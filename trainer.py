from AMGT.network import AssistBranch, InferenceBranch
from utils import *
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

    def run_network(self, inputs, mask, labels, model):
        fine_logits, coarse_logits = model(inputs)
        fused_logits = fine_logits * self.args.fine_weight + coarse_logits * (
            1 - self.args.fine_weight
        )
        fused_probs = F.sigmoid(fused_logits) * mask.unsqueeze(
            2
        )  # (B, T, n_class)
        coarse_loss = self.criterion(coarse_logits, labels) / torch.sum(mask)
        fine_loss = self.criterion(fine_logits, labels) / torch.sum(mask)
        loss = coarse_loss + fine_loss * self.args.fine_weight
        return fused_logits, fused_probs, loss

    def train_step(self, data, model, apm):

        inputs, mask, labels, other, hm = data

        self.optimizer.zero_grad()
        model.train()

        fused_logits, fused_probs, loss = self.run_network(
            inputs, mask, labels, model
        )
        # 4. backward pass
        loss.backward()
        self.optimizer.step()
        # 5. Metric update
        apm.add(fused_probs.data.cpu().numpy()[0], labels.cpu().numpy()[0])

        return (fused_logits, loss, fused_probs)

    def eval_step(self, data, model, apm, sampled_apm, full_probs):

        inputs, mask, labels, other, hm = data

        model.eval()
        with torch.no_grad():
            fused_logits, fused_probs, loss = self.run_network(
                inputs, mask, labels, model
            )
            # Metric update
            apm.add(fused_probs.data.cpu().numpy()[0], labels.cpu().numpy()[0])
            if sum(inputs.numpy()[0]) > 25:
                p1, l1 = sampled_25(
                    fused_probs.data.cpu().numpy()[0],
                    labels.cpu().numpy()[0],
                    mask=mask.cpu().numpy()[0],
                )
                sampled_apm.add(p1, l1)

            full_probs[other[0][0]] = mask_probs(
                fused_probs.data.cpu().numpy()[0],
                mask.cpu().numpy()[0].squeeze(),
            ).T

        return loss

    def epoch_metrics_updates(self, apm, loss, num_iter):
        map = 100 * apm.value().mean()
        apm.reset()
        epoch_loss = loss / num_iter
        return map, epoch_loss

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
                _, loss, _ = self.train_step(
                    inputs, mask, labels, self.assist, a_apm
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
                _, loss, _ = self.train_step(
                    inputs, mask, labels, self.inference, i_apm
                )
                i_loss += loss.item()
            # train epoch level metrics update
            a_map, a_epoch_loss = self.epoch_metrics_updates(
                a_apm, a_loss, num_iter
            )
            i_map, i_epoch_loss = self.epoch_metrics_updates(
                i_apm, i_loss, num_iter
            )
            print(
                f"Assist Branch - train-map: {a_map:.4f}, train-loss: {a_epoch_loss:.4f}"
            )
            print(
                f"Inference Branch - train-map: {i_map:.4f}, train-loss: {i_epoch_loss:.4f}"
            )
            epoch_end = time()
            print(
                f"Training Epoch time: {(epoch_end - epoch_start)/60:.2f} mins"
            )

            epoch_start = time()
            # Evaluate Epoch Level
            apm = APMeter()
            sampled_apm = APMeter()
            v_loss = 0.0
            full_probs = {}
            num_iter = 0
            for batch in tqdm(val_loader, desc="Validation"):
                num_iter += 1
                # unpack batch
                inputs, mask, labels, other, hm = batch
                inputs = inputs.to(self.args.device)  # (B, T, D)
                mask = mask.to(self.args.device) if mask is not None else None
                labels = labels.to(self.args.device)  # (B, T, n_class)
                # inference eval step level
                loss = self.eval_step(
                    (inputs, mask, labels, other, hm),
                    self.inference,
                    apm,
                    sampled_apm,
                    full_probs,
                )
                v_loss += loss.item()
            # val epoch level metrics update
            val_map, val_epoch_loss = self.epoch_metrics_updates(
                apm, v_loss, num_iter
            )
            sampled_val_map, _ = self.epoch_metrics_updates(
                sampled_apm, v_loss, num_iter
            )
            print(
                f"Validation-map: {val_map:.4f}, sampled-25%-map: {sampled_val_map:.4f}, val-loss: {val_epoch_loss:.4f}"
            )
            epoch_end = time()
            print(
                f"Validation Epoch time: {(epoch_end - epoch_start)/60:.2f} mins"
            )

            if val_map > Best_val_map:
                Best_val_map = val_map
                torch.save(
                    self.inference.state_dict(),
                    f"{self.args.checkpoint_path}/best_model.pth",
                )
                print("Best model saved.")
