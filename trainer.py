from AMGT.network import AssistBranch, InferenceBranch

import torch


def trainning_step(batch, assist, inference, optimizer, criterion, device):
    """
    Perform a training step.

    Args:
        batch: A tuple containing input features and ground truth labels.
        assist: AssistBranch model.
        inference: InferenceBranch model.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device: Device to run the computations on.

    Returns:
        loss: Computed loss for the batch.
    """
    inputs, gt_labels = batch
    inputs = inputs.to(device)  # (B, T, D)
    gt_labels = gt_labels.to(device)  # (B, T, n_class)

    optimizer.zero_grad()

    # Forward pass through Assist Branch
    assist_outputs = assist(gt_labels)  # (B, T, n_class)

    # Forward pass through Inference Branch
    inference_outputs = inference(inputs)  # (B, T, n_class)

    # Compute loss
    loss_assist = criterion(
        assist_outputs.view(-1, assist_outputs.size(-1)),
        gt_labels.view(-1, gt_labels.size(-1)),
    )
    loss_inference = criterion(
        inference_outputs.view(-1, inference_outputs.size(-1)),
        gt_labels.view(-1, gt_labels.size(-1)),
    )
    loss = loss_assist + loss_inference

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    return loss.item()
