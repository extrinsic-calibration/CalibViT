import torch
from torch import nn as nn
import numpy as np 
import transform
from typing import Tuple

class QuaternionLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(QuaternionLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.losses = {}

    def forward(self, target_transl, target_rot, transl_err, rot_err)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean() * 100
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = transform.quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot

        return total_loss, loss_rot, loss_transl 