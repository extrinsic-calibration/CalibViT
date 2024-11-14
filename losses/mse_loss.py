import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MSETransformationLoss(nn.Module):
    """
    Calculate the MSE loss for rotation and translation components of transformation matrices.
    """
    def __init__(self):
        super(MSETransformationLoss, self).__init__()
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, Tcl: torch.Tensor, igt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute the MSE loss for translation and rotation.

        Args:
            Tcl (torch.Tensor): The cumulative predicted transformation matrix, shape [B, 4, 4].
            igt (torch.Tensor): The inverse ground truth transformation matrix, shape [B, 4, 4].

        Returns:
            torch.Tensor: MSE loss for translation.
            torch.Tensor: MSE loss for rotation.
        """
        # Invert igt to get the actual ground truth transformation
        true_transformation = torch.inverse(igt)
        error =  Tcl.bmm(igt)
        
        # Extract translation vectors (last column but exclude the final row)
        pred_translation = Tcl[:, :3, 3]
        true_translation = true_transformation[:, :3, 3]
        
        # Calculate MSE loss for translation
        translation_loss = F.mse_loss(pred_translation, true_translation)
       
        # Extract rotation matrices (top-left 3x3 sub matrix)
        pred_rotation = Tcl[:, :3, :3]
        true_rotation = true_transformation[:, :3, :3]
        
        # Calculate MSE loss for rotation matrices directly
        rotation_loss = rotation_loss = F.mse_loss(pred_rotation, true_rotation) 
        return translation_loss, rotation_loss

    @staticmethod      
    def rotation_loss(R_pred, R_gt):
        # Calculate the trace for each pair of rotation matrices in the batch
        batch_trace = torch.einsum('bij,bij->b', R_pred, R_gt.transpose(-2, -1))
        # Clamp to prevent values outside of [-1, 1] due to numerical precision issues
        trace_value = torch.clamp((batch_trace - 1) / 2, -1.0, 1.0)
        # Compute the mean of the arccosine for the rotation loss
        return torch.mean(torch.acos(trace_value))
    
    @staticmethod
    def translation_loss(t_pred, t_gt):
         return torch.mean(torch.norm(t_pred - t_gt, dim=1) ** 2)