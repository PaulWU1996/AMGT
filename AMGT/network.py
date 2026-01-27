import torch
import torch.nn as nn
import torch.nn.functional as F

from AMGT.transformer import EncoderLayer, ShawRPE
from AMGT.msbm import BiMamba2, DSBlock, MultiScaleBiMamba, Mixture
from AMGT.cls import Classifier


class AssistBranch(nn.Module):

    def __init__(self, d_model, n_class, n_layers=2, n_head=8, max_offset=512):
        """
        Assist Branch Module for Guided Training.

        Input: GT (B, T, n_class)


        """

        super().__init__()

        self.proj = nn.Linear(n_class, d_model)
        self.rpe = ShawRPE(
            dim=d_model, max_offset=max_offset, bidirectional=True
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_q=d_model,
                    d_kv=d_model,
                    d_head=d_model // n_head,
                    n_head=n_head,
                    rpe=self.rpe,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        self.classifier = Classifier(d_model=d_model, n_cls=n_class)

    def forward(self, gt_onehot, mask=None):
        # gt_onehot: (B, T, n_class)

        x = self.proj(gt_onehot)  # (B, T, D)

        enc_out = x
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, mask=mask)  # (B, T, D)

        enc_out = self.norm(enc_out)  # (B, T, D)
        coarse_probs, fine_probs = self.classifier(
            enc_out
        )  # (B, T, n_class), (B, T, n_class)

        return coarse_probs, fine_probs, enc_out  # (B, T, D)


class InferenceBranch(nn.Module):

    def __init__(
        self,
        d_model,
        scale_factor=2,
        depth=3,
        d_state=64,
        d_conv=4,
        expand=2,
        mode="linear",
        align_corners=True,
        n_cls=32,
    ):

        super().__init__()

        self.downsampler = DSBlock(
            d_model=d_model, scale_factor=scale_factor, depth=depth
        )

        self.msbm = MultiScaleBiMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            depth=depth,
        )

        self.mixture = Mixture(
            d_model=d_model, depth=depth, mode=mode, align_corners=align_corners
        )

        self.classifier = Classifier(d_model=d_model, n_cls=n_cls)

    def forward(
        self,
        x,
    ):
        # x: (B, T, D)

        downsampled_feats = self.downsampler(x)  # List of (B, T//s, D)

        msbm_out = self.msbm(x, downsampled_feats)  # (B, T, D)

        mixed_feat = self.mixture(msbm_out, downsampled_feats)  # (B, T, D)

        coarse_probs, fine_probs = self.classifier(
            mixed_feat
        )  # (B, T, n_class), (B, T, n_class)

        return coarse_probs, fine_probs  # (B, T, n_class), (B, T, n_class)
