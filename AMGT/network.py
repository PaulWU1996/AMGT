import torch
import torch.nn as nn
import torch.nn.functional as F

from AMGT.transformer import EncoderLayer, ShawRPE
from AMGT.msbm import BiMamba2, DSBlock, MultiScaleBiMamba, Mixture
from AMGT.cls import Classifier


class AssistBranch(nn.Module):
    def __init__(self, d_model, n_class, n_layers=2, n_head=8, max_offset=512, dropout=0.1):
        """
        Assist Branch Module for Guided Training.
        该分支通过处理 GT One-hot 序列，为模型提供时序上下文引导。
        """
        super().__init__()
        
        # 1. 输入投影：将 label 空间映射到特征空间
        self.proj = nn.Linear(n_class, d_model)
        
        # 2. 实例化共享 RPE
        self.d_head = d_model // n_head
        self.rpe = ShawRPE(
            d_head=self.d_head, 
            max_offset=max_offset, 
            bidirectional=True
        )

        # 3. 契合优化后脚本的 EncoderLayers
        # 确保传入了 rpe 实例，内部 Attention 会自动处理 einsum 点积
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,  # 对应 Pre-LN 脚本中的参数名
                    d_kv=d_model,
                    d_head=self.d_head,
                    n_head=n_head,
                    dropout=dropout,
                    rpe=self.rpe,
                )
                for _ in range(n_layers)
            ]
        )
        
        # Pre-LN 架构需要在所有层结束后进行最终归一化
        self.norm = nn.LayerNorm(d_model)

        # 假设 Classifier 接收特征并输出细粒度/粗粒度预测
        self.classifier = Classifier(d_model=d_model, n_cls=n_class)

    def forward(self, gt_onehot, mask=None):
        """
        Args:
            gt_onehot (torch.Tensor): (B, T, n_class)
            mask (torch.Tensor, optional): (B, T) 时序掩码
        """
        # 初始线性投影
        x = self.proj(gt_onehot)
        
        # 逐层经过 Encoder (Self-Attention 模式)
        enc_out = x
        for layer in self.encoder_layers:
            # 在自注意力模式下，x_q = x_kv = enc_out
            enc_out = layer(enc_out, enc_out, mask=mask)

        # 最终归一化
        enc_out = self.norm(enc_out)
        
        # 分类器输出
        # 根据你的代码，这里传入了两个 enc_out，可能内部做特征融合或 query-based 检测
        fine_logits, coarse_logits = self.classifier(enc_out, enc_out)
        
        return fine_logits, coarse_logits


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
        fine_weight=0.1,
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

        self.classifier = Classifier(
            d_model=d_model, n_cls=n_cls, fine_weight=fine_weight
        )

    def forward(self, x):

        downsampled_feats = self.downsampler(x)
        msbm_out = self.msbm(downsampled_feats)
        fine_feat, coarse_feat = self.mixture(msbm_out)
        fine_feat, coarse_feat = self.classifier(coarse_feat, fine_feat)

        return fine_feat, coarse_feat
