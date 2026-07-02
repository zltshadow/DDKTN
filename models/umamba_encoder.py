"""
U-Mamba Encoder for DDKTN.
Adapted from U-Mamba (https://github.com/bowang-lab/U-Mamba).
Provides a 2D Mamba-based encoder that produces multi-scale feature maps.
"""
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple
from torch.amp import autocast


# ---------------------------------------------------------------------------
# Mamba SSM Layer
# ---------------------------------------------------------------------------
class MambaLayer(nn.Module):
    """Mamba Selective State Space Model layer.

    Wraps the mamba_ssm.Mamba module. Handles spatial feature maps by
    flattening them into token sequences.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        from mamba_ssm import Mamba
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    @autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        if self.channel_token:
            return self.forward_channel_token(x)
        else:
            return self.forward_patch_token(x)


# ---------------------------------------------------------------------------
# Basic Residual Block
# ---------------------------------------------------------------------------
class BasicResBlock(nn.Module):
    """Basic residual block with InstanceNorm and LeakyReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1,
                 use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


# ---------------------------------------------------------------------------
# Residual Mamba Encoder (from U-Mamba)
# ---------------------------------------------------------------------------
class ResidualMambaEncoder2D(nn.Module):
    """2D encoder with Mamba layers at selected stages.

    Adapted from U-Mamba's ResidualMambaEncoder for 2D medical image
    segmentation. Produces multi-scale feature maps as skip connections.
    """
    def __init__(self, input_channels=1, n_stages=6,
                 features_per_stage=(32, 64, 128, 256, 320, 320),
                 kernel_sizes=((3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)),
                 strides=((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
                 n_blocks_per_stage=(2, 2, 2, 2, 2, 2),
                 input_size=(384, 384)):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages

        # Always use patch_token mode (channel_token=False).
        # channel_token mode bakes spatial dimensions into Mamba's d_model,
        # which breaks when runtime input size differs from construction-time
        # input_size (e.g. due to random scale augmentation).  Patch token
        # mode uses features_per_stage as d_model — size-independent.
        do_channel_token = [False] * n_stages
        feature_map_size = list(input_size)
        feature_map_sizes = []
        for s in range(n_stages):
            feature_map_sizes.append(
                [i // j for i, j in zip(feature_map_size, strides[s])]
            )
            feature_map_size = feature_map_sizes[-1]

        self.conv_pad_sizes = [[k // 2 for k in krnl] for krnl in kernel_sizes]

        # Stem
        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(input_channels, stem_channels,
                          kernel_size=kernel_sizes[0], padding=self.conv_pad_sizes[0],
                          use_1x1conv=True),
            *[BasicResBlock(stem_channels, stem_channels,
                            kernel_size=kernel_sizes[0], padding=self.conv_pad_sizes[0])
              for _ in range(n_blocks_per_stage[0] - 1)]
        )

        # Encoder stages
        stages = []
        mamba_layers = []
        in_ch = stem_channels
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(in_ch, features_per_stage[s],
                              kernel_size=kernel_sizes[s],
                              padding=self.conv_pad_sizes[s],
                              stride=strides[s],
                              use_1x1conv=True),
                *[BasicResBlock(features_per_stage[s], features_per_stage[s],
                                kernel_size=kernel_sizes[s],
                                padding=self.conv_pad_sizes[s])
                  for _ in range(n_blocks_per_stage[s] - 1)]
            )
            # Insert Mamba at alternating stages (guarantee last stage has Mamba)
            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_layers.append(
                    MambaLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s]
                            else features_per_stage[s],
                        channel_token=do_channel_token[s]
                    )
                )
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            in_ch = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.output_channels = list(features_per_stage)
        self.strides = [list(s) for s in strides]

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            skips.append(x)
        return skips


# ---------------------------------------------------------------------------
# U-Mamba Encoder Wrapper for DDKTN
# ---------------------------------------------------------------------------
class UMambaEncoder2D(nn.Module):
    """U-Mamba encoder wrapper for DDKTN.

    Produces multi-scale feature maps and a coarse segmentation output
    from the deepest features via a lightweight segmentation head.

    Args:
        input_channels: Number of input channels (e.g., 1 for grayscale CT)
        num_classes: Number of segmentation classes
        features_per_stage: Feature dimensions at each encoder stage
        strides: Downsampling strides at each stage
        n_blocks_per_stage: Number of conv blocks per stage
        input_size: Input spatial size (H, W)
    """
    def __init__(self, input_channels=1, num_classes=4,
                 features_per_stage=(32, 64, 128, 256, 320, 320),
                 strides=((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
                 n_blocks_per_stage=(2, 2, 2, 2, 1, 1),
                 kernel_sizes=((3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)),
                 input_size=(384, 384)):
        super().__init__()
        n_stages = len(features_per_stage)
        self.encoder = ResidualMambaEncoder2D(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            input_size=input_size,
        )
        # Lightweight segmentation head from deepest features
        self.seg_head = nn.Sequential(
            nn.Conv2d(features_per_stage[-1], 256, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image
        Returns:
            skips: list of multi-scale feature maps
            coarse_logits: (B, num_classes, H', W') coarse segmentation
        """
        skips = self.encoder(x)
        coarse_logits = self.seg_head(skips[-1])
        return skips, coarse_logits

    @property
    def output_channels(self):
        return self.encoder.output_channels

    @property
    def strides(self):
        return self.encoder.strides


# ---------------------------------------------------------------------------
# nnUNet-style Decoder for U-Mamba (full encoder-decoder)
# ---------------------------------------------------------------------------
class _PlainConvBlock(nn.Module):
    """Plain double 3×3 conv block (no residual shortcut), matching nnUNet default."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UMambaDecoder2D(nn.Module):
    """nnUNet-style decoder that mirrors the U-Mamba encoder.

    Takes multi-scale skip connections from the encoder and produces
    full-resolution segmentation logits via transposed convolution
    upsampling and double-conv blocks (matching nnUNet's default decoder).

    Supports deep supervision: when deep_supervision=True, returns
    auxiliary logits at 2× and 4× downsampled scales (from the two
    deepest decoder stages), in addition to the full-resolution output.

    Args:
        features_per_stage: Feature dimensions at each encoder stage
        num_classes: Number of output segmentation classes
        n_blocks_per_stage: Number of conv blocks per decoder stage
        deep_supervision: If True, return auxiliary outputs for deep supervision
    """
    def __init__(self, features_per_stage=(32, 64, 128, 256, 512, 512),
                 num_classes=4, n_blocks_per_stage=(2, 2, 2, 2, 2, 2),
                 deep_supervision=False):
        super().__init__()
        n = len(features_per_stage)
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # Decoder stages: from deepest (n-1) back to stage 0
        # At each level: upsample + concat skip → conv blocks
        self.ups = nn.ModuleList()
        self.stages = nn.ModuleList()

        for i in range(n - 2, -1, -1):
            ch_hi = features_per_stage[i + 1]
            ch_lo = features_per_stage[i]
            # Transposed convolution for 2× upsampling
            self.ups.append(nn.ConvTranspose2d(ch_hi, ch_hi, kernel_size=2, stride=2))
            # After concatenation: input channels = ch_hi + ch_lo
            blocks = [_PlainConvBlock(ch_hi + ch_lo, ch_lo)]
            for _ in range(n_blocks_per_stage[i] - 1):
                blocks.append(_PlainConvBlock(ch_lo, ch_lo))
            self.stages.append(nn.Sequential(*blocks))

        # Final 1×1 conv for multi-class logits
        self.final_conv = nn.Conv2d(features_per_stage[0], num_classes, 1)

        # Deep supervision heads (auxiliary classifiers at deeper decoder stages)
        if deep_supervision:
            # heads for the two deepest decoder stages (index 0 and 1)
            # these output at 2× and 4× downsampled resolution
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(features_per_stage[n - 2], num_classes, 1),
                nn.Conv2d(features_per_stage[n - 3], num_classes, 1),
            ])

    def forward(self, skips):
        """
        Args:
            skips: list of multi-scale feature maps from encoder
                    (stage 0 = finest, stage n-1 = deepest)
        Returns:
            If deep_supervision=False:
                logits: (B, num_classes, H, W)
            If deep_supervision=True:
                [logits, aux2, aux4] where aux2/aux4 are at 2×/4× downsampled
        """
        x = skips[-1]
        ds_outputs = []

        for i, (up, stage) in enumerate(zip(self.ups, self.stages)):
            skip_idx = len(skips) - 2 - i
            x = up(x)
            # Handle size mismatch from stride/padding differences
            if x.shape[2:] != skips[skip_idx].shape[2:]:
                x = F.interpolate(x, size=skips[skip_idx].shape[2:],
                                  mode='bilinear', align_corners=False)
            x = torch.cat([x, skips[skip_idx]], dim=1)
            x = stage(x)

            # Collect deep supervision outputs from the two deepest stages
            if self.deep_supervision and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))

        logits = self.final_conv(x)

        if self.deep_supervision:
            return [logits] + ds_outputs
        return logits


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def build_umamba_encoder(input_channels=3, num_classes=21,
                         features_per_stage=(32, 64, 128, 256, 512, 512),
                         strides=((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
                         n_blocks_per_stage=(2, 2, 2, 2, 2, 2),
                         input_size=(256, 256),
                         deep_supervision=False):
    """Build a full U-Mamba encoder-decoder for standalone training.

    Returns a model whose forward(x) returns (B, num_classes, H, W) logits
    at full input resolution (or a list of logits if deep_supervision=True).
    """
    encoder = ResidualMambaEncoder2D(
        input_channels=input_channels,
        n_stages=len(features_per_stage),
        features_per_stage=features_per_stage,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
        input_size=input_size,
    )
    decoder = UMambaDecoder2D(
        features_per_stage=features_per_stage,
        num_classes=num_classes,
        n_blocks_per_stage=n_blocks_per_stage,
        deep_supervision=deep_supervision,
    )
    return _UMambaSegModel(encoder, decoder)


class _UMambaSegModel(nn.Module):
    """Thin wrapper combining encoder + decoder for standalone training."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
