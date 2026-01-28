import torch
import torch.nn as nn

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb is not installed. WandbLogger will not be available.")


class AsymetricLoss(nn.Module):

    def __init__(
        self,
        gamma_neg=3,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """

        Args:
            x : input logits
            y : targets (multi-label binarized vector)
        """

        # Calculate probabilities
        x_sigmoid = torch.sigmoid(x)
        x_pos = x_sigmoid
        x_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            x_neg = (x_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(x_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(x_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt = x_pos * y + x_neg * (1 - y)
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


def log(content, epoch=None, step=None, prefix="training"):

    if epoch is None and step is None:
        raise ValueError("Either epoch or step must be provided for logging.")

    # WandB Logger
    if WANDB_AVAILABLE and wandb.run is not None:
        if epoch is not None:
            wandb.log({**content, "epoch": epoch})
        elif step is not None:
            wandb.log({**content, "global_step": step})

    # Txt Logger
    else:

        if isinstance(content, dict):

            log_str = " | ".join(
                [
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in content.items()
                ]
            )

        else:
            log_str = str(content)
        if epoch is not None:
            filename = f"{prefix}_epoch.txt"
            with open(filename, "a") as f:
                f.write(f"Epoch {epoch}: {log_str}\n")
        if step is not None:
            filename = f"{prefix}_step.txt"
            with open(filename, "a") as f:
                f.write(f"Step {step}: {log_str}\n")
