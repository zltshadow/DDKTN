"""
Dynamic Feature Interaction (DFI) Module
Implements gated cross-attention with temporal adaptive weights for
bidirectional knowledge transfer between SAM2 and U-Mamba branches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedCrossAttention(nn.Module):
    """Gated Cross-Attention mechanism for DFI block.

    Computes cross-attention between two branch features with a
    head-specific sigmoid gate to suppress irrelevant cross-branch noise.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections for cross-attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Sigmoid gate parameters
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, query_feat, key_value_feat):
        """
        Args:
            query_feat: (B, C, H, W) - features from the query branch
            key_value_feat: (B, C, H, W) - features from the key/value branch
        Returns:
            out: (B, C, H, W) - cross-attended features
        """
        B, C, H, W = query_feat.shape

        # Flatten spatial dimensions: (B, H*W, C)
        q = query_feat.flatten(2).transpose(1, 2)
        kv = key_value_feat.flatten(2).transpose(1, 2)

        # Project Q, K, V
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # Reshape for multi-head attention
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (memory-efficient via FlashAttention)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Reshape back: (B, H*W, C)
        out = out.transpose(1, 2).contiguous().view(B, H * W, C)
        out = self.out_proj(out)

        # Apply sigmoid gate using query branch features
        gate_input = q.transpose(1, 2).contiguous().view(B, H * W, C)
        # Use pre-norm hidden state of query branch for gating
        gate = torch.sigmoid(self.gate_proj(gate_input))
        out = gate * out

        # Reshape back to spatial: (B, C, H, W)
        out = out.transpose(1, 2).view(B, C, H, W)

        return out


class ChannelSpatialAlign(nn.Module):
    """Channel and spatial alignment module to project features from
    different branches into a common latent space before interaction.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_align = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)

    def forward(self, x, target_size=None):
        """
        Args:
            x: (B, C_in, H, W)
            target_size: (H_target, W_target) for spatial alignment
        Returns:
            aligned: (B, C_out, H_target, W_target)
        """
        x = self.channel_align(x)
        x = self.norm(x)
        if target_size is not None and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class DFIBlock(nn.Module):
    """Dynamic Feature Interaction Block.

    Implements bidirectional cross-attention between SAM2 and U-Mamba
    features at each encoder stage, with temporal adaptive weights.

    The interaction is governed by:
        F_SAM' = F_SAM + alpha'_SAM←UM * Gated(H_SAM, Attn_S←UM)
        F_UM' = F_UM + alpha'_UM←SAM * Gated(H_UM, Attn_U←S)

    where alpha' is determined by a sigmoid function incorporating
    temporal decay and performance disparity Delta_P_t.
    """
    def __init__(self, sam_channels, um_channels, align_channels=256, num_heads=8):
        super().__init__()
        # Channel alignment
        self.align_sam = ChannelSpatialAlign(sam_channels, align_channels)
        self.align_um = ChannelSpatialAlign(um_channels, align_channels)

        # Cross-attention: SAM queries U-Mamba
        self.cross_attn_s_from_u = GatedCrossAttention(align_channels, num_heads)
        # Cross-attention: U-Mamba queries SAM
        self.cross_attn_u_from_s = GatedCrossAttention(align_channels, num_heads)

        # Learnable residual projections
        self.residual_sam = nn.Conv2d(align_channels, align_channels, 1)
        self.residual_um = nn.Conv2d(align_channels, align_channels, 1)

        self.align_channels = align_channels

    def compute_dynamic_weight(self, t, T_max, delta_p_t=0.0):
        """Compute temporal adaptive weight alpha'.

        Args:
            t: current iteration step
            T_max: total iteration steps
            delta_p_t: performance disparity (ASD difference)
        Returns:
            alpha_sam_from_um: weight for U-Mamba -> SAM2 knowledge flow
            alpha_um_from_sam: weight for SAM2 -> U-Mamba knowledge flow
        """
        # Temporal decay term + performance correction
        temporal = (t / T_max) - 0.5
        # Sigmoid to bound in (0, 1)
        alpha_sam_from_um = torch.sigmoid(torch.tensor(temporal + delta_p_t))
        alpha_um_from_sam = 1.0 - alpha_sam_from_um
        return alpha_sam_from_um, alpha_um_from_sam

    def forward(self, f_sam, f_um, t=0, T_max=100000, delta_p_t=0.0):
        """
        Args:
            f_sam: (B, C_sam, H, W) - SAM2 branch features at stage l
            f_um: (B, C_um, H, W) - U-Mamba branch features at stage l
            t: current training iteration
            T_max: total training iterations
            delta_p_t: performance disparity from ASD
        Returns:
            f_sam_out: (B, align_channels, H, W) - refined SAM2 features
            f_um_out: (B, align_channels, H, W) - refined U-Mamba features
        """
        # Determine target spatial size (use SAM2's resolution)
        target_size = f_sam.shape[2:]

        # Align channels and spatial dimensions
        f_sam_aligned = self.align_sam(f_sam, target_size)  # (B, C_align, H, W)
        f_um_aligned = self.align_um(f_um, target_size)     # (B, C_align, H, W)

        # Compute dynamic weights
        alpha_s_from_u, alpha_u_from_s = self.compute_dynamic_weight(
            t, T_max, delta_p_t
        )

        # Cross-attention: SAM queries U-Mamba
        attn_s = self.cross_attn_s_from_u(f_sam_aligned, f_um_aligned)
        # Cross-attention: U-Mamba queries SAM
        attn_u = self.cross_attn_u_from_s(f_um_aligned, f_sam_aligned)

        # Apply dynamic weights and residual connection
        f_sam_out = f_sam_aligned + alpha_s_from_u * attn_s
        f_um_out = f_um_aligned + alpha_u_from_s * attn_u

        return f_sam_out, f_um_out


class MultiScaleDFI(nn.Module):
    """Multi-scale DFI that applies DFIBlock at each encoder stage.

    Manages feature alignment between SAM2 (uniform 256ch) and
    U-Mamba (variable 32/64/128/256/320/320ch) at multiple scales.
    """
    def __init__(self, sam_channels_list, um_channels_list, align_channels=256, num_heads=8):
        super().__init__()
        self.num_stages = len(sam_channels_list)

        # Create DFI blocks for each stage
        self.dfi_blocks = nn.ModuleList()
        for i in range(self.num_stages):
            self.dfi_blocks.append(
                DFIBlock(
                    sam_channels=sam_channels_list[i],
                    um_channels=um_channels_list[i],
                    align_channels=align_channels,
                    num_heads=num_heads
                )
            )

    def forward(self, sam_features, um_features, t=0, T_max=100000, delta_p_t=0.0):
        """
        Args:
            sam_features: list of (B, C_sam_l, H_l, W_l) from SAM2 encoder stages
            um_features: list of (B, C_um_l, H_l, W_l) from U-Mamba encoder stages
            t, T_max, delta_p_t: temporal parameters
        Returns:
            refined_sam: list of refined SAM2 features at each stage
            refined_um: list of refined U-Mamba features at each stage
        """
        refined_sam = []
        refined_um = []

        for i in range(self.num_stages):
            s_out, u_out = self.dfi_blocks[i](
                sam_features[i], um_features[i], t, T_max, delta_p_t
            )
            refined_sam.append(s_out)
            refined_um.append(u_out)

        return refined_sam, refined_um
