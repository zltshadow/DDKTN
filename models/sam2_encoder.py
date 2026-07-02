"""
SAM2 Encoder Wrapper for DDKTN.
Adapted from SAM 2 (https://github.com/facebookresearch/sam2).
Wraps the SAM2 Hiera image encoder and prompt encoder/mask decoder
for integration into the dual-branch DDKTN framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SAM2EncoderWrapper(nn.Module):
    """SAM2 image encoder wrapper for DDKTN.

    Extracts multi-scale features from SAM2's Hiera backbone + FPN neck,
    and provides access to the prompt encoder and mask decoder for
    prompt-based refinement.

    Args:
        sam2_model: A built SAM2Base model (from build_sam2)
        freeze_backbone: Whether to freeze the image encoder weights
    """
    def __init__(self, sam2_model, freeze_backbone=True):
        super().__init__()
        self.image_encoder = sam2_model.image_encoder
        self.prompt_encoder = sam2_model.sam_prompt_encoder
        self.mask_decoder = sam2_model.sam_mask_decoder

        if freeze_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def encode_image(self, image_tensor):
        """Encode image through SAM2's Hiera backbone + FPN neck.

        Args:
            image_tensor: (B, 3, 1024, 1024) normalized image tensor
        Returns:
            vision_features: (B, 256, H, W) top-level features
            backbone_fpn: list of (B, 256, H_l, W_l) multi-scale features
            vision_pos_enc: list of positional encodings
        """
        output = self.image_encoder(image_tensor)
        return output["vision_features"], output["backbone_fpn"], output["vision_pos_enc"]

    def decode_masks(self, vision_features, vision_pos_enc, high_res_features,
                     point_coords=None, point_labels=None, boxes=None,
                     mask_input=None, image_size=1024):
        """Decode masks using SAM2's prompt encoder + mask decoder.

        Args:
            vision_features: (B, 256, H, W) from image encoder
            vision_pos_enc: positional encodings
            high_res_features: list of high-res backbone features
            point_coords: (B, N, 2) point prompts
            point_labels: (B, N) point labels
            boxes: (B, 4) box prompts
            mask_input: (B, 1, 256, 256) low-res mask input
            image_size: original image size
        Returns:
            masks: (B, 1, H, W) predicted masks
            iou_predictions: (B,) IoU scores
            low_res_masks: (B, 1, 256, 256) low-res masks
        """
        B = vision_features.shape[0]
        device = vision_features.device

        # Prepare sparse and dense embeddings from prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels) if point_coords is not None else None,
            boxes=boxes,
            masks=mask_input,
        )

        # high_res_features should be [feat_s0, feat_s1]
        # expected:
        #   feat_s0: [B, 32, 256, 256]
        #   feat_s1: [B, 64, 128, 128]

        if high_res_features is not None and self.mask_decoder.use_high_res_features:
            feat_s0, feat_s1 = high_res_features[:2]

            # If they are still 256-channel SAM/DFI features, project them
            if feat_s0.shape[1] == 256:
                feat_s0 = self.mask_decoder.conv_s0(feat_s0)  # 256 -> 32

            if feat_s1.shape[1] == 256:
                feat_s1 = self.mask_decoder.conv_s1(feat_s1)  # 256 -> 64

            high_res_features = [feat_s0, feat_s1]

        # Decode masks
        image_pe = self.prompt_encoder.get_dense_pe()
        image_pe = image_pe.to(device=vision_features.device, dtype=vision_features.dtype)

        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=vision_features,
            image_pe=image_pe,                      # 关键：用 get_dense_pe()
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,                     # batched images
            high_res_features=high_res_features,
        )

        # Upscale to original resolution
        masks = F.interpolate(
            low_res_masks, (image_size, image_size),
            mode="bilinear", align_corners=False,
        )

        return masks, iou_predictions, low_res_masks

    def forward(self, image_tensor, point_coords=None, point_labels=None,
                boxes=None, mask_input=None):
        """Full forward pass: encode image + decode masks.

        Args:
            image_tensor: (B, 3, 1024, 1024) normalized image
            point_coords: (B, N, 2) optional point prompts
            point_labels: (B, N) optional point labels
            boxes: (B, 4) optional box prompts
            mask_input: (B, 1, 256, 256) optional mask input
        Returns:
            masks: (B, 1, H, W) predicted masks
            iou_predictions: (B,) IoU scores
            backbone_fpn: list of multi-scale features for DFI
        """
        vision_features, backbone_fpn, vision_pos_enc = self.encode_image(image_tensor)

        # Use last two backbone features as high-res features
        high_res_features = backbone_fpn[:2] if len(backbone_fpn) >= 2 else backbone_fpn

        masks, iou_preds, low_res_masks = self.decode_masks(
            vision_features, vision_pos_enc, high_res_features,
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
            image_size=image_tensor.shape[2],
        )

        return masks, iou_preds, backbone_fpn
