"""
MSMB: Multi-Scale Bi-Mamba for Action Detection.

This module provides UNET-similar Mamba framework for action detection.
It includes classes and functions to use.

Author: Dr. Peipei (Paul) Wu
Date: Jan 2026
Contact: peipei.wu@surrey.ac.uk
Copyright: University of Surrey
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2


class ResTemporalDS(nn.Module):

    def __init__(self, d_model, scale=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=scale + 1,
                stride=scale,
                padding=scale // 2,
                groups=d_model,
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.residual = nn.AvgPool1d(kernel_size=scale, stride=scale)

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        out = self.conv(x)  # (B, D, T//scale)
        out = out + self.residual(x)  # (B, D, T//scale)
        out = out.transpose(1, 2)  # (B, T//scale, D)
        return out


class DSBlock(nn.Module):

    def __init__(self, d_model, scale_factor=2, depth=3):
        super().__init__()
        scales = [scale_factor**i for i in range(1, depth)]
        self.layers = nn.ModuleList()
        for scale in scales:
            self.layers.append(ResTemporalDS(d_model, scale=scale))

    def forward(self, x):
        out = [x]
        if len(self.layers) == 0:
            return out
        for layer in self.layers:
            out.append(layer(x))
        return out


class BiMamba2(nn.Module):
    """Bi-directional Mamba2 Block"""

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()

        self.mamba_fwd = Mamba2(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.mamba_bwd = Mamba2(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x: (B, T, D)
        out_fwd = self.mamba_fwd(x)

        x_flip = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_flip)
        out_bwd = torch.flip(out_bwd, dims=[1])

        combined = torch.cat([out_fwd, out_bwd], dim=-1)
        return self.norm(x + self.proj(combined))


class MultiScaleBiMamba(nn.Module):

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, depths=3):
        """

        Args:
            depths (int, optional): Number of scales. Defaults to 3.

            Other Args follow BiMamba2.
        """
        super().__init__()
        self.encoders = nn.ModuleList()
        for _ in range(depths):
            self.encoders.append(
                BiMamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )

    def forward(self, X):
        # X: List of multi-scale features [(B, T, D), (B, T/2, D), (B, T/4, D), ...]
        out = []
        for i, x in enumerate(X):
            m_out = self.encoders[i](x)  # (B, T//(scale_factor**i), D)
            out.append(m_out)  # (B, T//(scale_factor**i), D)
        return out  # List of (B, T, D) for diverse scales T: T, T/2, T/4, ...


class Mixture(nn.Module):
    """Mixture Module to combine multi-scale features"""

    def __init__(self, d_model, depths=3, mode="linear", align_corners=True):
        super().__init__()

        self.mode = mode
        self.align_corners = align_corners
        self.projs = nn.ModuleList()
        for _ in range(depths):
            self.projs.append(nn.Linear(d_model, d_model))

    def resize(self, x, target_t, mode="linear"):
        # x: (B, Ts, D)
        x = x.transpose(1, 2)  # (B, D, Ts)

        if mode == "linear":
            x = F.interpolate(
                x,
                size=target_t,
                mode="linear",
                align_corners=self.align_corners,
            )
        elif mode == "nearest":
            x = F.interpolate(x, size=target_t, mode="nearest")
        else:
            raise ValueError(f"Unsupported resize mode: {mode}")
        x = x.transpose(1, 2)  # (B, T, D)
        return x

    def forward(self, X):
        # X: List of multi-scale features [(B, T, D), (B, T/2, D), (B, T/4, D), ...]
        _, T, _ = X[0].size()

        fine, coarse = 0, 0
        for i, x in enumerate(X):
            x = self.projs[i](x)
            if i == 0:
                fine = x
            else:
                coarse += self.resize(x, T, mode=self.mode)
        return fine, coarse
