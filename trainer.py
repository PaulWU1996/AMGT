from AMGT.network import AssistBranch, InferenceBranch

import torch


def trainning_step(batch, assist, inference, optimizer_assist, optimizer_inference, criterion_assist, criterion_inference, device):
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
    inputs, gt_labels = batch
    inputs = inputs.to(device)  # (B, T, D)
    gt_labels = gt_labels.to(device)  # (B, T, n_class)
    optimizer_assist.zero_grad()
    optimizer_inference.zero_grad()

    # Step 1: Assist Branch Forward Pass
    assist_outputs = assist(gt_labels)  # (B, T, n_class)
    loss_assist = criterion_assist(assist_outputs, gt_labels)
    loss_assist.backward()
    optimizer_assist.step()

    # Step 2: Copy Classifier Weights from Assist to Inference Branch and Freeze
    inference.classifier.load_state_dict(assist.classifier.state_dict())
    for param in inference.classifier.parameters():
        param.requires_grad = False

    # Step 3: Inference Branch Forward Pass
    inference_outputs = inference(inputs)  # (B, T, n_class)
    loss_inference = criterion_inference(inference_outputs, gt_labels)
    loss_inference.backward()
    optimizer_inference.step()

    

