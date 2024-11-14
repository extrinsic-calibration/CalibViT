import torch
import torch.nn.functional as F
from torch import nn

class PhotoLoss(nn.Module):
    def __init__(self, scale=1.0, reduction='mean', loss_type='Charbonnier') ->  torch.Tensor:
        super(PhotoLoss, self).__init__()
        assert reduction in ['sum', 'mean', 'none'], 'Unknown or invalid reduction'
        self.scale = scale
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Photo loss

        Args:
            source (torch.Tensor): (B, 3, H, W)
            target (torch.Tensor): (B, 3, H, W)

        Returns:
            torch.Tensor: scaled mse loss between input and target
        """
        # Scale the inputs
        if self.loss_type == 'L1':
            return nn.functional.l1_loss(source, target)
        elif self.loss_type == 'L2':
            return nn.functional.mse_loss(source, target)
        elif self.loss_type == 'Charbonnier':
            epsilon = 1e-6
            return torch.mean(torch.sqrt((target - source) ** 2 + epsilon ** 2))
        else:
            loss = F.mse_loss(source, target, reduction=self.reduction)
            return loss

    def __call__(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(source, target)