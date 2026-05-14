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


# class ResTemporalDS(nn.Module):

#     def __init__(self, d_model, scale=2):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=d_model,
#                 out_channels=d_model,
#                 kernel_size=scale + 1,
#                 stride=scale,
#                 padding=scale // 2,
#                 groups=d_model,
#             ),
#             nn.BatchNorm1d(d_model),
#             nn.GELU(),
#         )
#         self.residual = nn.AvgPool1d(kernel_size=scale, stride=scale)

#     def forward(self, x):
#         # x: (B, T, D)
#         x = x.transpose(1, 2)  # (B, D, T)
#         out = self.conv(x)  # (B, D, T//scale)
#         out = out + self.residual(x)  # (B, D, T//scale)
#         out = out.transpose(1, 2)  # (B, T//scale, D)
#         return out


# class DSBlock(nn.Module):

#     def __init__(self, d_model, scale_factor=2, depth=3):
#         super().__init__()
#         scales = [scale_factor**i for i in range(1, depth)]
#         self.layers = nn.ModuleList()
#         for scale in scales:
#             self.layers.append(ResTemporalDS(d_model, scale=scale))

#     def forward(self, x):
#         out = [x]
#         if len(self.layers) == 0:
#             return out
#         for layer in self.layers:
#             out.append(layer(x))
#         return out


# class BiMamba2(nn.Module):
#     """Bi-directional Mamba2 Block"""

#     def __init__(self, d_model, d_state=128, d_conv=4, expand=2):
#         super().__init__()

#         self.mamba_fwd = Mamba2(
#             d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
#         )

#         self.mamba_bwd = Mamba2(
#             d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
#         )

#         self.norm = nn.LayerNorm(d_model)
#         self.proj = nn.Linear(d_model * 2, d_model)

#     def forward(self, x):
#         # x: (B, T, D)
#         out_fwd = self.mamba_fwd(x)

#         x_flip = torch.flip(x, dims=[1])
#         out_bwd = self.mamba_bwd(x_flip)
#         out_bwd = torch.flip(out_bwd, dims=[1])

#         combined = torch.cat([out_fwd, out_bwd], dim=-1)
#         return self.norm(x + self.proj(combined))


# class MultiScaleBiMamba(nn.Module):

#     def __init__(self, d_model, d_state=64, d_conv=4, expand=2, depth=3):
#         """

#         Args:
#             depths (int, optional): Number of scales. Defaults to 3.

#             Other Args follow BiMamba2.
#         """
#         super().__init__()
#         self.encoders = nn.ModuleList()
#         for _ in range(depth):
#             self.encoders.append(
#                 BiMamba2(
#                     d_model=d_model,
#                     d_state=d_state,
#                     d_conv=d_conv,
#                     expand=expand,
#                 )
#             )

#     def forward(self, X):
#         # X: List of multi-scale features [(B, T, D), (B, T/2, D), (B, T/4, D), ...]
#         out = []
#         for i, x in enumerate(X):
#             m_out = self.encoders[i](x)  # (B, T//(scale_factor**i), D)
#             out.append(m_out)  # (B, T//(scale_factor**i), D)
#         return out  # List of (B, T, D) for diverse scales T: T, T/2, T/4, ...


# class Mixture(nn.Module):
#     """Mixture Module to combine multi-scale features"""

#     def __init__(self, d_model, depth=3, mode="linear", align_corners=True):
#         super().__init__()

#         self.mode = mode
#         self.align_corners = align_corners
#         self.projs = nn.ModuleList()
#         for _ in range(depth):
#             self.projs.append(nn.Linear(d_model, d_model))

#     def resize(self, x, target_t, mode="linear"):
#         # x: (B, Ts, D)
#         x = x.transpose(1, 2)  # (B, D, Ts)

#         if mode == "linear":
#             x = F.interpolate(
#                 x,
#                 size=target_t,
#                 mode="linear",
#                 align_corners=self.align_corners,
#             )
#         elif mode == "nearest":
#             x = F.interpolate(x, size=target_t, mode="nearest")
#         else:
#             raise ValueError(f"Unsupported resize mode: {mode}")
#         x = x.transpose(1, 2)  # (B, T, D)
#         return x

#     def forward(self, X):
#         # X: List of multi-scale features [(B, T, D), (B, T/2, D), (B, T/4, D), ...]
#         _, T, _ = X[0].size()

#         fine, coarse = 0, 0
#         for i, x in enumerate(X):
#             x = self.projs[i](x)
#             if i == 0:
#                 fine = x
#             else:
#                 coarse += self.resize(x, T, mode=self.mode)
#         return fine, coarse


class NBiMamba2(nn.Module):
    """Bi-directional Mamba2 Block"""

    def __init__(self, d_model, d_state=128, d_conv=4, expand=2):
        super().__init__()


        self.mamba_fwd = Mamba2(
            d_model=d_model * 2, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.mamba_bwd = Mamba2(
            d_model=d_model * 2, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.pre_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
        )

        self.post_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: (B, T, D)
        shortcut = x
        x = self.pre_proj(x)  # (B, T, 2D)
        x_flip = torch.flip(x, dims=[1]) # (B, T, 2D)
        gate = torch.nn.functional.silu(x) 

        out_fwd = self.mamba_fwd(x)  # (B, T, 2D)
        out_bwd = self.mamba_bwd(x_flip)  # (B, T, 2D)
        out_bwd = torch.flip(out_bwd, dims=[1])  # (B, T, 2D)

        out_fwd = out_fwd * gate
        out_bwd = out_bwd * gate

        out = self.post_proj(out_fwd + out_bwd)  # (B, T, D)
        return out + shortcut # (B, T, D)

class MSBM(nn.Module):
    def __init__(
        self,
        d_input,
        d_embed,
        d_state=128,
        d_conv=4,
        expand=2,
        scales=[2, 4], 
    ):
        super().__init__()
        self.scales = scales
        
        self.downsamples = nn.ModuleList()
        for s in scales:
            self.downsamples.append(
                Temporal(in_channels=d_input, out_channels=d_embed, stride=s)
            )
            
        self.encoders = nn.ModuleList()
        for s in scales:
            current_state = d_state + (s * 16) 
            self.encoders.append(
                NBiMamba2(
                    d_model=d_embed,
                    d_state=current_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )


    def forward(self, x):
        # x (B, T, D)
        B, T, D = x.shape
        multi_scale_out = []

        x_t = x.transpose(1, 2)

        for i in range(len(self.scales)):
            # 1. (B, D, T) -> (B, D, T//s)
            s_feat = self.downsamples[i](x_t)
            
            # 2. (B, T//s, D) 
            s_feat = s_feat.transpose(1, 2)
            s_out = self.encoders[i](s_feat)
            
            # 3. (T//s -> T)
            if s_out.size(1) != T:
                s_out_aligned = torch.nn.functional.interpolate(
                    s_out.transpose(1, 2), 
                    size=T, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
                multi_scale_out.append(s_out_aligned)
            else:
                multi_scale_out.append(s_out)
        
        return torch.sum(torch.stack(multi_scale_out, dim=0), dim=0)  # (B, T, D) 
        # return multi_scale_out  # List of (B, T, D) for each scale

class Temporal(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=256,
        kernel_size=3,
        stride=1,
    ):

        super().__init__()
        if stride == 1:
            padding = kernel_size // 2
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            )
        else:
            kernel_size = stride + 1 if stride % 2 == 0 else stride
            padding = (kernel_size - stride + 1) // 2
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            )
        self.norm = nn.LayerNorm(out_channels)
        # self.apply(self._init_weights)


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, D, T)
        x = self.conv(x)  # (B, D_out, T) stride=1 时保持 T 不变，stride=2 时变为 T//2
        x = x.transpose(1, 2)  # (B, T, D_out)
        x = self.norm(x)  # (B, T, D_out)
        x = x.transpose(1, 2)  # (B, D_out, T)
        return x
    
class MSFusion(nn.Module):
    def __init__(
        self,
        d_model,
        scales=3,
        ):
        super().__init__()

        self.proj = nn.Conv1d(d_model * scales, d_model, kernel_size=1)
        self.fusion_mamba = NBiMamba2(d_model=d_model)

    def forward(self, x):
        # x: (B, S, T, D)
        B, S, T, D = x.shape
        x_ = x.transpose(1, 2).reshape(B, T, S * D)  # (B, T, S*D)
        x_ = self.proj(x_.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        fused = self.fusion_mamba(x_)  # (B, T, D)
        
        fused_norm = F.normalize(fused, dim=-1)  # (B, T, D)
        attn_temporal = torch.bmm(fused_norm, fused_norm.transpose(1, 2))  # (B, T, T)

        return fused, attn_temporal