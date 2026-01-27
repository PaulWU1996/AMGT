"""
Transformer: Relative Positional Attention based Transformer for Action Detection.

This module provides UNET-similar Mamba framework for action detection.
It includes classes and functions to use.

Author: Dr. Peipei (Paul) Wu
Date: Jan 2026
Contact: peipei.wu@surrey.ac.uk
"""

from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShawRPE(nn.Module):

    def __init__(
        self,
        dim: int,
        max_offset: int = 512,
        max_len: int = 2048,
        bidirectional: bool = True,
    ):
        """

        Args:
            dim (int): Embedding dimension. Defaults to 2048.
            max_offset (int): Maximum relative position range.
            max_len (int, optional): Maximum embedding length. Defaults to 2048.
            bidirectional (bool, optional): Single or bi-direction. Defaults to True.
        """
        super(ShawRPE, self).__init__()

        self.dim = dim
        self.max_offset = max_offset
        self.max_len = max_len
        if bidirectional:
            self.pos_emb = nn.Embedding(2 * max_offset + 1, dim)
        else:
            self.pos_emb = nn.Embedding(max_offset + 1, dim)
        self.register_buffer(
            "rel_index",
            self.build_relative_position(max_len, max_offset, bidirectional),
            persistent=False,
        )

    @staticmethod
    def build_relative_position(max_len, max_offset, bidirectional):
        pos = torch.arange(max_len)
        rel = pos[None, :] - pos[:, None]  # shape: [L, L]
        if bidirectional:
            rel = (
                rel.clamp(-max_offset, max_offset) + max_offset
            )  # shift to [0, 2*max_offset]
        else:
            rel = rel.clamp(0, max_offset)  # shift to [0, max_offset]
        return rel.long()

    def forward(self, seq_len):
        """

        Args:
            seq_len (int): The length of the sequence.

        Returns:
            nn.Tensor : The relative positional encoding tensor of shape [seq_len, seq_len, dim].
        """
        assert (
            seq_len <= self.max_len
        ), f"Input sequence length {seq_len} exceeds the maximum embedding length {self.max_len}"

        index = self.rel_index[:seq_len, :seq_len]  # shape: [seq_len, seq_len]
        return self.pos_emb(index)  # shape: [seq_len, seq_len, dim]


class Attention(nn.Module):

    def __init__(self, d_q: int, d_kv: int, d_head: int, n_head: int, **kwargs):
        super(Attention, self).__init__()
        self.W_q = nn.Linear(d_q, d_head * n_head)
        self.W_k = nn.Linear(d_kv, d_head * n_head)
        self.W_v = nn.Linear(d_kv, d_head * n_head)
        self.scale = d_head ** (-0.5)
        self.n_head = n_head
        self.d_head = d_head
        self.W_o = nn.Linear(d_head * n_head, d_q)

        if kwargs.get("share_norm", True) and d_q == d_kv:
            norm = nn.LayerNorm(d_q)
            self.norm_q = norm
            self.norm_kv = norm
        else:
            self.norm_q = nn.LayerNorm(d_q)
            self.norm_kv = nn.LayerNorm(d_kv)

        self.rpe = kwargs.get("rpe", None)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Optional[torch.Tensor] = None,
    ):

        assert (
            q.dim() == 3 and kv.dim() == 3
        ), "输入必须是 3 维张量 [batch, seq_len, dim]"

        B, N_q, _ = q.size()
        B, N_kv, _ = kv.size()

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Step 1:  [B, n_head, N, d_head]
        Q = self.W_q(q).view(B, N_q, self.n_head, self.d_head).transpose(1, 2)
        K = self.W_k(kv).view(B, N_kv, self.n_head, self.d_head).transpose(1, 2)
        V = self.W_v(kv).view(B, N_kv, self.n_head, self.d_head).transpose(1, 2)

        # Step 2:
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Step 3: TorchScript Support
        if self.rpe is not None:
            # rpe_bias shape: [N_kv, N_kv, d_head]
            rpe_bias = self.rpe(N_kv).to(Q.device)

            # Q:  [N_q, B * n_head, d_head]
            q_rpe = Q.permute(2, 0, 1, 3).reshape(
                N_q, B * self.n_head, self.d_head
            )
            # rpe_bias: [N_kv, d_head, N_kv]
            r_rpe = rpe_bias.transpose(1, 2)

            # bmm : [N_q, B * n_head, N_kv]
            rpe_scores = torch.bmm(q_rpe, r_rpe)

            # reshape to [B, n_head, N_q, N_kv]
            rpe_scores = rpe_scores.reshape(N_q, B, self.n_head, N_kv).permute(
                1, 2, 0, 3
            )

            scores = scores + rpe_scores * self.scale

        # Step 4: Mask and Softmax
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == False, float("-inf"))

        atten = F.softmax(scores, dim=-1)

        # Step 5: 输出投影
        out = torch.matmul(atten, V)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, N_q, self.n_head * self.d_head)
        )
        return self.W_o(out), atten


class EncoderLayer(nn.Module):

    def __init__(self, d_q: int, d_kv: int, d_head: int, n_head: int, **kwargs):
        """

        Args:
            d_q (int): Dimension of query.
            d_kv (int): Dimension of key and value.
            d_head (int): Dimension of each head.
            n_head (int): Number of attention heads.

            **kwargs: Additional arguments for Attention and Feed-Forward Network.
        """
        super(EncoderLayer, self).__init__()

        self.attn = Attention(
            d_q=d_q, d_kv=d_kv, d_head=d_head, n_head=n_head, **kwargs
        )
        self.norm = nn.LayerNorm(d_q)
        self.ffn = kwargs.get(
            "ffn",
            nn.Sequential(
                nn.Linear(d_q, 4 * d_head * n_head),
                nn.GELU(),
                nn.Linear(4 * d_head * n_head, d_q),
            ),
        )

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask=None):
        """

        Args:
            x_q (torch.Tensor): Query tensor [B, N_q, D_q]
            x_kv (torch.Tensor): Key/Value tensor [B, N_kv, D_kv]
            mask (_type_, optional): _description_. Defaults to None.

        When x_q and x_kv are the same, it performs self-attention; otherwise, it performs cross-attention.
        """

        x_q = x_q + self.attn(q=x_q, kv=x_kv, mask=mask)[0]
        x_q = self.norm(x_q)
        x_q = x_q + self.ffn(x_q)
        return x_q


class DecoderLayer(nn.Module):

    def __init__(self, d_q: int, d_kv: int, d_head: int, n_head: int, **kwargs):
        """

        Args:
            d_q (int): Dimension of query.
            d_kv (int): Dimension of key and value.
            d_head (int): Dimension of each head.
            n_head (int): Number of attention heads.

            **kwargs: Additional arguments for Attention and Feed-Forward Network.
        """
        super(DecoderLayer, self).__init__()

        self.self_attn = Attention(
            d_q=d_q, d_kv=d_q, d_head=d_head, n_head=n_head, **kwargs
        )

        self.cross_attn = Attention(
            d_q=d_q, d_kv=d_kv, d_head=d_head, n_head=n_head, **kwargs
        )

        self.norm = nn.LayerNorm(d_q)

        self.ffn = kwargs.get(
            "ffn",
            nn.Sequential(
                nn.Linear(d_q, 4 * d_head * n_head),
                nn.GELU(),
                nn.Linear(4 * d_head * n_head, d_q),
            ),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        self_mask=None,
        cross_mask=None,
    ):
        """

        Args:
            tgt (torch.Tensor): Decoder input embeddings [B, N_tgt, D_q]
            memory (torch.Tensor): Encoder output used as key/value in cross-attention [B, N_mem, D_kv]
            self_mask (_type_, optional): Mask tensor for self-attention. Defaults to None.
            cross_mask (_type_, optional): Mask tensor for cross-attention. Defaults to None.
        """

        tgt = tgt + self.self_attn(tgt, tgt, self_mask)[0]
        tgt = tgt + self.cross_attn(tgt, memory, cross_mask)[0]
        tgt = self.norm(tgt)
        tgt = tgt + self.ffn(tgt)
        return tgt


class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_q: int,
        d_kv: int,
        d_head: int,
        n_head: int,
        **kwargs,
    ):
        """

        Args:
            n_layers (int): Number of layers in Encoder/Decoder.
            d_q (int): Dimension of query.
            d_kv (int): Dimension of key and value.
            d_head (int): Dimension of each head.
            n_head (int): Number of attention heads.

            **kwargs: Additional arguments for Attention and Feed-Forward Network.
        """

        super(Transformer, self).__init__()

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    d_q=d_q, d_kv=d_kv, d_head=d_head, n_head=n_head, **kwargs
                )
                for _ in range(n_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    d_q=d_q, d_kv=d_kv, d_head=d_head, n_head=n_head, **kwargs
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        src_q: torch.Tensor,
        src_kv: torch.Tensor,
        tgt: torch.Tensor,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
    ):
        """

        Args:
            src_q (torch.Tensor): Encoder query tensor [B, N_q, D_q]
            src_kv (torch.Tensor): Encoder key/value tensor [B, N_kv, D_kv]
            tgt (torch.Tensor): Decoder input tensor [B, N_tgt, D_q]
            src_mask (_type_, optional): _description_. Defaults to None.
            tgt_mask (_type_, optional): _description_. Defaults to None.
            memory_mask (_type_, optional): _description_. Defaults to None.
        """

        # Encoder
        enc_out = src_q
        for layer in self.encoder:
            enc_out = layer(enc_out, src_kv, src_mask)

        # Decoder
        dec_out = tgt
        for layer in self.decoder:
            dec_out = layer(
                dec_out, enc_out, self_mask=tgt_mask, cross_mask=memory_mask
            )

        return dec_out
