"""
Transformer: Relative Positional Attention based Transformer for Action Detection.
Optimized for Pre-LN architecture and dynamic RPE integration.

Author: Dr. Peipei (Paul) Wu (Refined by AI Collaborator)
Date: Jan 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ShawRPE(nn.Module):
    def __init__(self, d_head: int, max_offset: int = 512, bidirectional: bool = True):
        super().__init__()
        self.d_head = d_head
        self.max_offset = max_offset
        self.bidirectional = bidirectional
        num_embeddings = (2 * max_offset + 1) if bidirectional else (max_offset + 1)
        self.pos_emb = nn.Embedding(num_embeddings, d_head)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def forward(self, n_q: int, n_kv: int, device: torch.device):
        range_q = torch.arange(n_q, device=device).view(-1, 1)
        range_kv = torch.arange(n_kv, device=device).view(1, -1)
        rel_index = range_q - range_kv

        if self.bidirectional:
            rel_index = torch.clamp(rel_index, -self.max_offset, self.max_offset) + self.max_offset
        else:
            rel_index = torch.clamp(rel_index, 0, self.max_offset)
        return self.pos_emb(rel_index)


class Attention(nn.Module):
    def __init__(self, d_q: int, d_kv: int, d_head: int, n_head: int, dropout: float = 0.1, rpe: Optional[nn.Module] = None):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.scale = d_head ** -0.5

        self.W_q = nn.Linear(d_q, d_head * n_head)
        self.W_k = nn.Linear(d_kv, d_head * n_head)
        self.W_v = nn.Linear(d_kv, d_head * n_head)
        self.W_o = nn.Linear(d_head * n_head, d_q)
        
        self.dropout = nn.Dropout(dropout)
        self.rpe = rpe

    def forward(self, q, kv, mask=None):
        B, N_q, _ = q.shape
        _, N_kv, _ = kv.shape

        Q = self.W_q(q).view(B, N_q, self.n_head, self.d_head).transpose(1, 2)
        K = self.W_k(kv).view(B, N_kv, self.n_head, self.d_head).transpose(1, 2)
        V = self.W_v(kv).view(B, N_kv, self.n_head, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))

        if self.rpe is not None:
            R = self.rpe(N_q, N_kv, Q.device)
            rpe_scores = torch.einsum('bhqd, qkd -> bhqk', Q, R)
            scores = scores + rpe_scores

        scores = scores * self.scale
        
        if mask is not None:
            # 自动处理 mask 维度广播
            if mask.dim() == 2: mask = mask[:, None, None, :]
            elif mask.dim() == 3: mask = mask[:, None, :, :]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, N_q, -1)
        return self.W_o(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_kv: int, d_head: int, n_head: int, dropout: float = 0.1, rpe: nn.Module = None):
        super().__init__()
        self.attn = Attention(d_model, d_kv, d_head, n_head, dropout, rpe)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x_q, x_kv, mask=None):
        # Pre-LN 结构
        res = x_q
        x_q, _ = self.attn(self.norm1(x_q), self.norm1(x_kv) if x_q is not x_kv else self.norm1(x_q), mask)
        x_q = res + x_q
        
        x_q = x_q + self.ffn(self.norm2(x_q))
        return x_q


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_kv: int, d_head: int, n_head: int, dropout: float = 0.1, rpe: nn.Module = None):
        super().__init__()
        self.self_attn = Attention(d_model, d_model, d_head, n_head, dropout, rpe)
        self.cross_attn = Attention(d_model, d_kv, d_head, n_head, dropout, rpe)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, tgt, memory, self_mask=None, cross_mask=None):
        # Self-Attention (Pre-LN)
        res = tgt
        tgt, _ = self.self_attn(self.norm1(tgt), self.norm1(tgt), self_mask)
        tgt = res + tgt
        
        # Cross-Attention (Pre-LN)
        res = tgt
        tgt, _ = self.cross_attn(self.norm2(tgt), memory, cross_mask) # memory 通常不需要 norm，因为它来自 encoder 已经输出了 norm
        tgt = res + tgt
        
        # FFN (Pre-LN)
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt


class Transformer(nn.Module):
    def __init__(self, n_layers: int, d_model: int, d_kv: int, n_head: int, max_offset: int = 512):
        super().__init__()
        d_head = d_model // n_head
        
        # 共享 RPE 实例
        self.rpe = ShawRPE(d_head, max_offset)
        
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, d_kv, d_head, n_head, rpe=self.rpe)
            for _ in range(n_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, d_kv, d_head, n_head, rpe=self.rpe)
            for _ in range(n_layers)
        ])

    def forward(self, src_q, src_kv, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_out = src_q
        for layer in self.encoder:
            enc_out = layer(enc_out, src_kv, src_mask)

        dec_out = tgt
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, tgt_mask, memory_mask)
        return dec_out