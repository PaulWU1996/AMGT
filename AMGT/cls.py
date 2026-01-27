"""
CLS: Classification for Action Detection.

This module provides classification framework for action detection.
It includes classes and functions to use.

Author: Dr. Peipei (Paul) Wu
Date: Jan 2026
Contact: peipei.wu@surrey.ac.uk
Copyright: University of Surrey
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, d_model, n_cls, fine_weight=0.1):
        super().__init__()

        self.fine_weight = fine_weight
        self.linear_coarse_1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.linear_coarse_2 = nn.Conv1d(d_model, n_cls, kernel_size=1)
        self.linear_fine = nn.Conv1d(d_model, n_cls, kernel_size=1)

        self.dropout = nn.Dropout()

    def forward(
        self,
        x_coarse,
        x_fine,
    ):
        # x_coarse: (B, T, D)
        # x_fine: (B, T, D)

        fine_probs = self.linear_fine(x_fine.transpose(1, 2))  # (B, n_cls, T)
        fine_probs = fine_probs.transpose(1, 2)  # (B, T, n_cls)

        x_coarse = self.linear_coarse_1(x_coarse.transpose(1, 2))  # (B, D, T)
        x_coarse = self.dropout(x_coarse)
        coarse_probs = self.linear_coarse_2(x_coarse)  # (B, n_cls, T)
        coarse_probs = coarse_probs.transpose(1, 2)  # (B, T, n_cls)

        fused_probs = (
            1 - self.fine_weight
        ) * coarse_probs + self.fine_weight * fine_probs
        return fused_probs
