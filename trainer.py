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

    def train_step(self, inputs, mask, labels, model):

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

        return (
            fused_logits,
            loss,
            fused_probs,
        )

    def fit(self, train_loader, val_loader):
        time_start = time()

        Best_val_map = 0.0

        for epoch in range(self.args.epochs):
            epoch_start = time()

            print(f"Epoch {epoch}/{self.args.epochs-1}")
            print("-" * 20)

            # Train Epoch Level
            for batch in train_loader:

                # unpack batch
                inputs, mask, labels, other, hm = batch
                inputs = inputs.to(self.args.device)  # (B, T, D)
                mask = mask.to(self.args.device) if mask is not None else None
                labels = labels.to(self.args.device)  # (B, T, n_class)

                # assist train step level
                a_logits, a_loss, a_probs = self.train_step(batch)

                # inference train step level

            # Evaluate Epoch Level


# def unpack_batch(batch, dataset, phase="training"):
#     inputs, mask, laebls, other, hm = batch
#     if dataset == "multithomus":
#         if phase == "training":
#             inputs = inputs.squeeze(3).squeeze(3)
#         else:
#             inputs = inputs.squeeze(0)
#             labels = labels.squeeze(dim=0)
#             if inputs.dim() == 5:
#                 inputs = inputs.squeeze(2).squeeze(2).permute(0, 2, 1)
#                 mask = mask.squeeze(0)
#             else:
#                 inputs = inputs.squeeze(1).squeeze(1).permute(1, 0).unsqueeze(0)
#                 labels = labels.unsqueeze(0)
#     elif dataset == "charades":
#         pass
#     else:
#         raise ValueError(f"Unknown dataset: {dataset}")


def train_step(
    args, dataloader, model, optimizer, criterion, device, epoch, network_type
):
    model.train()
    total_loss = 0.0
    num_iter = 0
    apm = APMeter()

    for batch in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        # Step 1: Unpack batch
        # TODO: link with unpack_batch to cover different datasets
        inputs, mask, labels, other, hm = batch
        inputs = inputs.to(device)  # (B, T, D)
        mask = mask.to(device) if mask is not None else None
        labels = labels.to(device)  # (B, T, n_class)

        # Step 2: Forward pass
        fine_logits, coarse_logits = model(inputs)

        # Step 3: Compute loss
        coarse_loss = criterion(coarse_logits, labels) / torch.sum(mask)
        fine_loss = criterion(fine_logits, labels) / torch.sum(mask)

        loss = coarse_loss + fine_loss * args.fine_weight

        # Step 4: Fused probs
        fused_probs = F.sigmoid(
            fine_logits * args.fine_weight
            + coarse_logits * (1 - args.fine_weight)
        ) * mask.unsqueeze(
            2
        )  # (B, T, n_class)

        # Step 5: APM Update
        apm.add(fused_probs.data.cpu().numpy()[0], batch[2].cpu().numpy()[0])
        total_loss += loss.item()

        # Step 6: Backward pass and optimization
        loss.backward()
        optimizer.step()

    train_map = 100 * apm.value().mean()
    if network_type is None:
        raise ValueError(
            "network_type must be specified as 'assist' or 'inference'"
        )
    print(
        "epoch", epoch, "network_type:", network_type, "train-map:", train_map
    )
    apm.reset()

    epoch_loss = total_loss / num_iter

    return train_map, epoch_loss


def evaluate_step(args, dataloader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    num_iter = 0
    apm = APMeter()
    sampled_apm = APMeter()

    for batch in tqdm(dataloader, desc="Evaluation"):
        num_iter += 1

        # Step 1: Unpack batch
        inputs, mask, labels, other, hm = batch
        inputs = inputs.to(device)  # (B, T, D)
        mask = mask.to(device) if mask is not None else None
        labels = labels.to(device)  # (B, T, n_class)

        # Step 2: Forward pass
        with torch.no_grad():
            fine_logits, coarse_logits = model(inputs)


def fit(
    args,
    dataloader,
    assist,
    inference,
    optimizer_assist,
    optimizer_inference,
    criterion,
    device,
):

    time_start = time()

    Best_val_map = 0.0

    for epoch in range(args.epochs):
        epoch_start = time()

        print(f"Epoch {epoch}/{args.epochs-1}")
        print("-" * 20)

        # Step 1: Train Assist Branch
        train_map_assist, train_loss_assist = train_step(
            args,
            dataloader,
            assist,
            optimizer_assist,
            criterion,
            device,
            epoch,
            network_type="assist",
        )

        # Step 2: Train Inference Branch
        # Copy classifier weights from Assist Branch to Inference Branch and freeze
        inference.classifier.load_state_dict(assist.classifier.state_dict())

        for param in inference.classifier.parameters():
            param.requires_grad = False
        for param in assist.classifier.parameters():
            param.requires_grad = True

        train_map_inference, train_loss_inference = train_step(
            args,
            dataloader,
            inference,
            optimizer_inference,
            criterion,
            device,
            epoch,
            network_type="inference",
        )
