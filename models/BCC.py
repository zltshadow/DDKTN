"""
Bidirectional Consensus Confidence (BCC) Module
Implements entropy-based consensus/divergence region partitioning
and dynamic mutual supervision for training stabilization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCCModule(nn.Module):
    """Bidirectional Consensus Confidence Module.

    Partitions the prediction space into:
    - Consensus regions R_C: both branches agree with high confidence
    - Divergence regions R_D: branches disagree, requiring mutual supervision

    Loss computation:
    - L_BCC = (1/|R_C|) * sum L_feat(x) + (1/|R_D|) * sum L_KD(x)

    In consensus regions: feature consistency loss (L2 norm)
    In divergence regions: entropy-based dynamic KL divergence
    """
    def __init__(self, tau_c=0.9, num_classes=4):
        """
        Args:
            tau_c: confidence threshold for consensus region (default 0.9)
            num_classes: number of segmentation classes (including background)
        """
        super().__init__()
        self.tau_c = tau_c
        self.num_classes = num_classes

    def compute_entropy(self, prob_map):
        """Compute prediction entropy H(x) = -sum_c P_c(x) * log(P_c(x)).

        Args:
            prob_map: (B, C, H, W) - probability map after softmax
        Returns:
            entropy: (B, 1, H, W) - per-pixel entropy
        """
        # Ensure float32 for numerical stability
        prob_map = prob_map.float()
        # Clamp to avoid log(0)
        prob_clamp = prob_map.clamp(min=1e-7)
        entropy = -(prob_clamp * torch.log(prob_clamp)).sum(dim=1, keepdim=True)
        return entropy

    def partition_regions(self, p_sam, p_um):
        """Partition pixels into consensus and divergence regions.

        Consensus: Y_SAM == Y_UM AND C_SAM > tau_c
        Divergence: Y_SAM != Y_UM

        Args:
            p_sam: (B, C, H, W) - SAM2 probability map
            p_um: (B, C, H, W) - U-Mamba probability map
        Returns:
            consensus_mask: (B, 1, H, W) - binary mask for consensus regions
            divergence_mask: (B, 1, H, W) - binary mask for divergence regions
        """
        # Predicted class and confidence
        y_sam = p_sam.argmax(dim=1, keepdim=True)  # (B, 1, H, W)
        y_um = p_um.argmax(dim=1, keepdim=True)    # (B, 1, H, W)
        c_sam = p_sam.max(dim=1, keepdim=True)[0]   # (B, 1, H, W)

        # Consensus: same prediction AND high confidence
        agree_mask = (y_sam == y_um).float()
        high_conf_mask = (c_sam > self.tau_c).float()
        consensus_mask = agree_mask * high_conf_mask

        # Divergence: different predictions
        divergence_mask = 1.0 - agree_mask

        return consensus_mask, divergence_mask

    def feature_consistency_loss(self, f_sam, f_um, mask):
        """L2 feature consistency loss in consensus regions.

        L_feat(x) = ||F'_SAM(x) - F'_UM(x)||_2^2

        Args:
            f_sam: (B, C, H, W) - SAM2 features after DFI
            f_um: (B, C, H, W) - U-Mamba features after DFI
            mask: (B, 1, H, W) - consensus region mask
        Returns:
            loss_feat: scalar - feature consistency loss
        """
        # Mean squared feature distance. Averaging over channels keeps this
        # term on a comparable scale when DFI uses wide feature maps.
        diff = (f_sam - f_um) ** 2
        diff = diff.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Apply mask and normalize
        masked_diff = diff * mask
        num_pixels = mask.sum().clamp(min=1.0)
        loss_feat = masked_diff.sum() / num_pixels

        return loss_feat

    def dynamic_kd_loss(self, p_sam, p_um, mask):
        """Jensen-Shannon divergence based knowledge distillation in divergence regions.

        Uses JS divergence instead of KL because:
        - JS is bounded in [0, log(2)] — prevents loss explosion
        - JS is symmetric — no need to designate teacher/student per pixel
        - Still transfers knowledge between branches

        Args:
            p_sam: (B, C, H, W) - SAM2 probability map
            p_um: (B, C, H, W) - U-Mamba probability map
            mask: (B, 1, H, W) - divergence region mask
        Returns:
            loss_kd: scalar - JS divergence loss
        """
        # Ensure float32 for numerical stability
        p_sam = p_sam.float()
        p_um = p_um.float()
        mask = mask.float()

        # Clamp probabilities for log computation
        eps = 1e-7
        p_sam_c = p_sam.clamp(min=eps)
        p_um_c = p_um.clamp(min=eps)

        # Mixture distribution M = 0.5 * (P + Q)
        m = 0.5 * (p_sam_c + p_um_c)

        # JS divergence: JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        # KL(P || M) = sum P * log(P / M)
        js_sam = (p_sam_c * (torch.log(p_sam_c) - torch.log(m))).sum(dim=1, keepdim=True)
        js_um = (p_um_c * (torch.log(p_um_c) - torch.log(m))).sum(dim=1, keepdim=True)
        js_loss = 0.5 * (js_sam + js_um)  # (B, 1, H, W), bounded in [0, log(2)]

        # Apply divergence mask and normalize
        masked_js = js_loss * mask
        num_pixels = mask.sum().clamp(min=1.0)
        loss_kd = masked_js.sum() / num_pixels

        return loss_kd

    def forward(self, p_sam, p_um, f_sam=None, f_um=None):
        """Compute BCC loss.

        Args:
            p_sam: (B, C, H, W) - SAM2 probability map
            p_um: (B, C, H, W) - U-Mamba probability map
            f_sam: (B, C_feat, H, W) - SAM2 features after DFI (for L_feat)
            f_um: (B, C_feat, H, W) - U-Mamba features after DFI (for L_feat)
        Returns:
            loss_bcc: scalar - total BCC loss
            consensus_mask: (B, 1, H, W) - for visualization
            divergence_mask: (B, 1, H, W) - for visualization
        """
        # Partition regions
        consensus_mask, divergence_mask = self.partition_regions(p_sam, p_um)

        # Feature consistency loss (if features provided)
        if f_sam is not None and f_um is not None:
            # Align spatial dimensions if needed
            if f_sam.shape[2:] != p_sam.shape[2:]:
                f_sam = F.interpolate(f_sam, size=p_sam.shape[2:], mode='bilinear', align_corners=False)
                f_um = F.interpolate(f_um, size=p_um.shape[2:], mode='bilinear', align_corners=False)
            loss_feat = self.feature_consistency_loss(f_sam, f_um, consensus_mask)
        else:
            loss_feat = torch.tensor(0.0, device=p_sam.device)

        # Dynamic KD loss
        loss_kd = self.dynamic_kd_loss(p_sam, p_um, divergence_mask)

        # Total BCC loss
        loss_bcc = loss_feat + loss_kd

        return loss_bcc, consensus_mask, divergence_mask
