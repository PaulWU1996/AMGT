from AMGT.network import AssistBranch, InferenceBranch
from utils import AsymetricLoss

import torch
from time import time
from tqdm import tqdm
import torch.nn.functional as F


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


def training_epoch(
    dataloader,
    assist,
    inference,
    optimizer_assist,
    optimizer_inference,
    criterion_assist,
    criterion_inference,
    device,
    epoch,
):
    total_loss_assist, total_loss_inference = 0.0, 0.0
    time_start = time()

    for batch in tqdm(dataloader, desc="Training Epoch {epoch}"):
        losses = trainning_step(
            dataset,
            batch,
            assist,
            inference,
            optimizer_assist,
            optimizer_inference,
            criterion_assist,
            criterion_inference,
            device,
        )

        total_loss_assist += losses["loss_assist"]
        total_loss_inference += losses["loss_inference"]
    time_end = time()
    avg_loss_assist = total_loss_assist / len(dataloader)
    avg_loss_inference = total_loss_inference / len(dataloader)
    print(f"Training Epoch completed in {time_end - time_start:.2f} seconds")
    return avg_loss_assist, avg_loss_inference


def trainning_step(
    dataset,
    batch,
    assist,
    inference,
    optimizer_assist,
    optimizer_inference,
    criterion_assist,
    criterion_inference,
    device,
):
    """

    Args:
        batch: _description_
        assist: _description_
        inference (_type_): _description_
        optimizer_assist (_type_): _description_
        optimizer_inference (_type_): _description_
        criterion_assist (_type_): _description_
        criterion_inference (_type_): _description_
        device (_type_): _description_
    """

    assist.train()
    inference.train()

    # TODO: link with unpack_batch to cover different datasets
    inputs, mask, labels, other, hm = batch

    inputs = inputs.to(device)  # (B, T, D)
    mask = mask.to(device) if mask is not None else None
    labels = labels.to(device)  # (B, T, n_class)

    optimizer_assist.zero_grad()
    optimizer_inference.zero_grad()

    # Step 1: Assist Branch Forward Pass
    assist_outputs = assist(labels)  # (B, T, n_class)
    assist_outputs = F.sigmoid(assist_outputs) * mask.unsqueeze(
        2
    )  # Apply mask if available
    loss_assist = criterion_assist(
        assist_outputs, labels
    )  # AsymetricLoss(assist_outputs, labels) -> logits (B, T, n_class), labels (B, T, n_class)

    loss_assist.backward()
    optimizer_assist.step()

    # Step 2: Copy Classifier Weights from Assist to Inference Branch and Freeze
    inference.classifier.load_state_dict(assist.classifier.state_dict())
    for param in inference.classifier.parameters():
        param.requires_grad = False

    for param in assist.classifier.parameters():
        param.requires_grad = True

    # Step 3: Inference Branch Forward Pass
    inference_outputs = inference(inputs)  # (B, T, n_class)
    loss_inference = criterion_inference(inference_outputs, gt_labels)
    loss_inference.backward()
    optimizer_inference.step()

    return {
        "loss_assist": loss_assist.item(),
        "loss_inference": loss_inference.item(),
    }


def evaluation_step(batch, inference, device, criterion, metric):

    inference.eval()

    inputs, gt_labels = batch
    inputs = inputs.to(device)  # (B, T, D)
    gt_labels = gt_labels.to(device)  # (B, T, n_class)

    with torch.no_grad():
        inference_outputs = inference(inputs)  # (B, T, n_class)

    loss = criterion(inference_outputs, gt_labels)
    metric(inference_outputs, gt_labels)
