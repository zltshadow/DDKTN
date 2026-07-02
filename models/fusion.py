"""
Fusion MLP Module
Generates final segmentation mask by fusing dual-branch features
via a Multilayer Perceptron.
"""
import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Fusion MLP for final segmentation prediction.

    Concatenates refined features from both SAM2 and U-Mamba branches
    after DFI processing, then generates the final probability map.

    P_final = MLP(Concat(F_P,SAM, F_P,UM))
    """
    def __init__(self, sam_channels, um_channels, num_classes, hidden_dim=256):
        """
        Args:
            sam_channels: channel dimension of SAM2 features after DFI
            um_channels: channel dimension of U-Mamba features after DFI
            num_classes: number of output segmentation classes
            hidden_dim: hidden layer dimension
        """
        super().__init__()
        input_channels = sam_channels + um_channels

        self.mlp = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GroupNorm(min(32, hidden_dim // 2), hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1)
        )

    def forward(self, f_sam, f_um):
        """
        Args:
            f_sam: (B, C_sam, H, W) - refined SAM2 features
            f_um: (B, C_um, H, W) - refined U-Mamba features
        Returns:
            logits: (B, num_classes, H, W) - segmentation logits
        """
        # Align spatial dimensions if needed
        if f_sam.shape[2:] != f_um.shape[2:]:
            f_um = nn.functional.interpolate(
                f_um, size=f_sam.shape[2:], mode='bilinear', align_corners=False
            )

        # Concatenate along channel dimension
        fused = torch.cat([f_sam, f_um], dim=1)

        # Generate logits
        logits = self.mlp(fused)

        return logits
