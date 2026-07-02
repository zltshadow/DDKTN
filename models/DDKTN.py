"""
DDKTN: Dual-Branch Dynamic Knowledge Transfer Network
Main model integrating SAM2 and U-Mamba branches with:
- PAM: Prompt Adaptation Module
- DFI: Dynamic Feature Interaction
- FusionMLP: Final prediction fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .umamba_encoder import UMambaEncoder2D, UMambaDecoder2D
from .sam2_encoder import SAM2EncoderWrapper
from .DFI import MultiScaleDFI
from .PAM import PAM
from .fusion import FusionMLP


class DDKTN(nn.Module):
    """Dual-Branch Dynamic Knowledge Transfer Network.

    Architecture:
        1. U-Mamba branch: Mamba-based encoder -> coarse segmentation + multi-scale features
        2. PAM: Generates box/point prompts from U-Mamba coarse predictions
        3. SAM2 branch: Hiera backbone -> multi-scale features; prompt encoder+mask decoder for refinement
        4. DFI: Bidirectional cross-attention between branches at each scale
        5. FusionMLP: Combines refined features from both branches

    Args:
        num_classes: Number of segmentation classes (including background)
        input_channels_um: Input channels for U-Mamba (1 for grayscale CT)
        input_size: Input spatial size (H, W) for U-Mamba
        sam2_model: Pre-built SAM2Base model instance
        um_features_per_stage: U-Mamba encoder feature dimensions
        sam2_backbone_channels: SAM2 backbone channel list (from config)
        dfi_align_channels: Channel dimension for DFI alignment
        dfi_num_heads: Number of attention heads in DFI
        pam_tau: PAM binarization threshold
        pam_bilateral: Whether PAM uses bilateral spatial decoupling
        freeze_sam2_backbone: Whether to freeze SAM2 image encoder
    """
    def __init__(
        self,
        num_classes=4,
        input_channels_um=1,
        input_size=(256, 256),
        sam2_model=None,
        um_features_per_stage=(32, 64, 128, 256, 512, 512),
        um_strides=((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
        um_n_blocks=(2, 2, 2, 2, 2, 2),
        sam2_backbone_channels=(576, 288, 144),
        dfi_align_channels=224,
        dfi_num_heads=8,
        pam_tau=0.9,
        pam_bilateral=True,
        freeze_sam2_backbone=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # ----- U-Mamba Branch -----
        self.umamba = UMambaEncoder2D(
            input_channels=input_channels_um,
            num_classes=num_classes,
            features_per_stage=um_features_per_stage,
            strides=um_strides,
            n_blocks_per_stage=um_n_blocks,
            input_size=input_size,
        )
        self.umamba_decoder = UMambaDecoder2D(
            features_per_stage=um_features_per_stage,
            num_classes=num_classes,
            n_blocks_per_stage=um_n_blocks,
        )

        # ----- SAM2 Branch -----
        assert sam2_model is not None, "A pre-built SAM2 model must be provided"
        self.sam2 = SAM2EncoderWrapper(
            sam2_model, freeze_backbone=freeze_sam2_backbone
        )

        # ----- DFI: Multi-scale Dynamic Feature Interaction -----
        # SAM2 FPN outputs all 256 channels; U-Mamba has variable channels
        # We align U-Mamba features at each scale to 256 via DFI
        num_dfi_stages = min(len(um_features_per_stage), len(sam2_backbone_channels))
        sam_channels_list = [256] * num_dfi_stages  # SAM2 FPN always outputs 256
        um_channels_list = list(um_features_per_stage[-num_dfi_stages:])

        self.dfi = MultiScaleDFI(
            sam_channels_list=sam_channels_list,
            um_channels_list=um_channels_list,
            align_channels=dfi_align_channels,
            num_heads=dfi_num_heads,
        )

        # ----- PAM: Prompt Adaptation Module -----
        self.pam = PAM(
            tau=pam_tau,
            num_classes=num_classes,
            bilateral=pam_bilateral,
        )

        # ----- Fusion MLP for final prediction -----
        # After DFI, both branches produce dfi_align_channels features
        self.fusion = FusionMLP(
            sam_channels=dfi_align_channels,
            um_channels=dfi_align_channels,
            num_classes=num_classes,
            hidden_dim=256,
        )
        # Learnable logit residuals keep the full-resolution branch decoders in
        # the main prediction path instead of using them only as auxiliary heads.
        self.fused_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.um_logit_scale = nn.Parameter(torch.tensor(0.0))
        self.sam_logit_scale = nn.Parameter(torch.tensor(0.0))

        # Projection layers to match DFI input from SAM2 backbone FPN
        self.sam_proj = nn.ModuleList([
            nn.Conv2d(256, 256, 1) for _ in range(num_dfi_stages)
        ])

        # SAM2 multi-class segmentation head (from backbone features + mask decoder)
        # Input: 256 backbone channels + 1 mask decoder binary mask = 257
        self.sam_seg_head = nn.Sequential(
            nn.Conv2d(257, 128, 3, padding=1),
            nn.InstanceNorm2d(128, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def _align_sam_features(self, backbone_fpn, num_stages):
        """Select and project SAM2 backbone features for DFI.

        SAM2 FPN produces features from coarsest to finest.
        We select the last num_stages features (finest) and project them.
        """
        selected = backbone_fpn[-num_stages:]
        aligned = []
        for i, feat in enumerate(selected):
            aligned.append(self.sam_proj[i](feat))
        return aligned

    def _align_um_features(self, um_skips, num_stages):
        """Select U-Mamba encoder features for DFI."""
        return um_skips[-num_stages:]

    def forward(self, image_um, image_sam2=None, t=0, T_max=100000, delta_p_t=0.0):
        """Forward pass of DDKTN.

        Args:
            image_um: (B, C_um, H, W) input for U-Mamba branch
            image_sam2: (B, 3, 1024, 1024) input for SAM2 branch (preprocessed)
                       If None, image_um is used (assumed already preprocessed for SAM2)
            t: Current training iteration (for DFI dynamic weights)
            T_max: Total training iterations
            delta_p_t: Performance disparity from ASD

        Returns:
            logits_fused: (B, num_classes, H, W) final segmentation
            logits_sam: (B, num_classes, H, W) SAM2 branch output
            logits_um: (B, num_classes, H, W) U-Mamba branch output (coarse)
            prob_um: (B, num_classes, H', W') U-Mamba probability for boundary loss
            bcc_info: dict with DFI features for BCC loss computation
        """
        if image_sam2 is None:
            image_sam2 = image_um

        B = image_um.shape[0]

        # ===== 1. U-Mamba Branch =====
        um_skips, coarse_logits = self.umamba(image_um)

        # Full-resolution U-Mamba logits from decoder
        logits_um = self.umamba_decoder(um_skips)
        prob_um = F.softmax(logits_um, dim=1)

        # ===== 2. PAM: Generate prompts from U-Mamba predictions =====
        # Generate box and point prompts
        point_coords, point_labels, box_coords = self.pam(prob_um, image_size=1024)
        point_coords = point_coords.to(image_um.device)
        point_labels = point_labels.to(image_um.device)
        box_coords = box_coords.to(image_um.device)

        # ===== 3. SAM2 Branch =====
        # Encode image
        vision_features, backbone_fpn, vision_pos_enc = self.sam2.encode_image(image_sam2)

        # Decode masks using prompts from PAM
        high_res_features = backbone_fpn[:2] if len(backbone_fpn) >= 2 else backbone_fpn
        sam_masks, iou_preds, low_res_masks = self.sam2.decode_masks(
            vision_features, vision_pos_enc, high_res_features,
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=box_coords,
            image_size=1024,
        )

        # ===== 4. Multi-scale DFI =====
        num_dfi_stages = len(self.dfi.dfi_blocks)
        sam_features = self._align_sam_features(backbone_fpn, num_dfi_stages)
        um_features = self._align_um_features(um_skips, num_dfi_stages)

        refined_sam, refined_um = self.dfi(
            sam_features, um_features,
            t=t, T_max=T_max, delta_p_t=delta_p_t,
        )

        # ===== 5. Fusion =====
        # Use the finest-scale refined features for fusion
        f_sam_final = refined_sam[-1]
        f_um_final = refined_um[-1]
        logits_fused = self.fusion(f_sam_final, f_um_final)
        # Upsample fused logits to match U-Mamba full resolution
        if logits_fused.shape[2:] != logits_um.shape[2:]:
            logits_fused = F.interpolate(
                logits_fused, size=logits_um.shape[2:],
                mode='bilinear', align_corners=False)

        # ===== 6. SAM2 branch logits (from backbone features + mask decoder) =====
        # Fuse backbone features with mask decoder output for multi-class prediction
        target_size = logits_um.shape[2:]
        sam_backbone_feat = backbone_fpn[-1]  # Finest scale backbone features (B, 256, H, W)
        # Resize mask decoder output to match backbone spatial dims
        sam_masks_resized = F.interpolate(
            sam_masks[:, 0:1],  # Take foreground mask (B, 1, H', W')
            size=sam_backbone_feat.shape[2:],
            mode='bilinear', align_corners=False,
        )
        # Concatenate: (B, 257, H, W)
        sam_enhanced = torch.cat([sam_backbone_feat, sam_masks_resized], dim=1)
        logits_sam = self.sam_seg_head(sam_enhanced)
        logits_sam = F.interpolate(
            logits_sam, size=target_size, mode='bilinear', align_corners=False
        )

        logits_fused = (
            self.fused_logit_scale * logits_fused
            + self.um_logit_scale * logits_um
            + self.sam_logit_scale * logits_sam
        )

        # ===== 7. Prepare BCC info =====
        bcc_info = {
            'f_sam': f_sam_final,
            'f_um': f_um_final,
            'refined_sam': refined_sam,
            'refined_um': refined_um,
        }

        return logits_fused, logits_sam, logits_um, prob_um, bcc_info

    def predict(self, image_um, image_sam2=None):
        """Inference-time prediction (no gradient computation).

        Args:
            image_um: (B, C_um, H, W) input for U-Mamba
            image_sam2: (B, 3, 1024, 1024) input for SAM2
        Returns:
            pred: (B, H, W) predicted segmentation mask
        """
        with torch.no_grad():
            logits_fused, _, _, _, _ = self.forward(image_um, image_sam2)
            pred = logits_fused.argmax(dim=1)
        return pred
