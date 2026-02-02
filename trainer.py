from AMGT.network import AssistBranch, InferenceBranch
from utils import AsymetricLoss
from apmeter import APMeter

import torch
from time import time
from tqdm import tqdm
import torch.nn.functional as F

""" 
- train-epoch-level

  - batch-level
    - unpack-batch
    - assist-train-step-level
        - forward-pass
        - logits-2-probs
        - compute-loss
        - backward-pass
    - inference-train-step-level

- val-epoch-level
    - batch-level
        - inference-eval-step-level
"""


class AMGT:

    def __init__(
        self,
        args,
        assist: AssistBranch,
        inference: InferenceBranch,
        optimizer,
        criterion,
    ):
        self.assist = assist
        self.inference = inference
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args

    def train_step(self, inputs, mask, labels, model, apm):

        self.optimizer.zero_grad()
        model.train()

        # 1. foward pass
        fine_logits, coarse_logits = model(inputs)
        # 2. logits to probs
        fused_logits = fine_logits * self.args.fine_weight + coarse_logits * (
            1 - self.args.fine_weight
        )
        fused_probs = F.sigmoid(fused_logits) * mask.unsqueeze(
            2
        )  # (B, T, n_class)
        # 3. compute loss
        coarse_loss = self.criterion(coarse_logits, labels) / torch.sum(mask)
        fine_loss = self.criterion(fine_logits, labels) / torch.sum(mask)
        loss = coarse_loss + fine_loss * self.args.fine_weight
        # 4. backward pass
        loss.backward()
        self.optimizer.step()
        # 5. Metric update
        apm.add(fused_probs.data.cpu().numpy()[0], labels.cpu().numpy()[0])

        return (fused_logits, loss, fused_probs)

    def fit(self, train_loader, val_loader):
        time_start = time()

        Best_val_map = 0.0

        for epoch in range(self.args.epochs):
            epoch_start = time()

            print(f"Epoch {epoch}/{self.args.epochs-1}")
            print("-" * 20)

            num_iter = 0
            a_loss = 0.0
            i_loss = 0.0
            a_apm = APMeter()
            i_apm = APMeter()

            # Train Epoch Level
            for batch in train_loader:

                num_iter += 1

                # unpack batch
                inputs, mask, labels, other, hm = batch
                inputs = inputs.to(self.args.device)  # (B, T, D)
                mask = mask.to(self.args.device) if mask is not None else None
                labels = labels.to(self.args.device)  # (B, T, n_class)

                # assist train step level
                (
                    a_logits,
                    loss,
                    a_probs,
                ) = self.train_step(
                    inputs,
                    mask,
                    labels,
                    self.assist,
                    a_apm,
                )
                a_loss += loss.item()

                # sync classifier weights and freeze
                self.inference.classifier.load_state_dict(
                    self.assist.classifier.state_dict()
                )
                for param in self.inference.classifier.parameters():
                    param.requires_grad = False
                for param in self.assist.classifier.parameters():
                    param.requires_grad = True

                # inference train step level
                i_logits, loss, i_probs = self.train_step(
                    inputs, mask, labels, self.inference, i_apm
                )
                i_loss += loss.item()
            # train epoch level metrics update
            a_map = 100 * a_apm.value().mean()
            i_map = 100 * i_apm.value().mean()
            a_apm.reset()
            i_apm.reset()
            a_epoch_loss = a_loss / num_iter
            i_epoch_loss = i_loss / num_iter
            print(
                f"Assist Branch - train-map: {a_map:.4f}, train-loss: {a_epoch_loss:.4f}"
            )
            print(
                f"Inference Branch - train-map: {i_map:.4f}, train-loss: {i_epoch_loss:.4f}"
            )
            # Evaluate Epoch Level
