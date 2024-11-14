from ast import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import torch
from torch import nn
from typing import Tuple

class Regression(nn.Module):
  
    def __init__(self, in_channels: int = 512, fc_hidden_layer_size: int = 128, dim: int = 256, 
                 fc_layer_dropout: float = 0.0) -> None:
        """
        Initializes the Regression module.

        Args:
            in_channels (int): The number of input channels (default is 512).
            fc_hidden_layer_size (int): The size of the hidden layers in the fully connected layers (default is 128).
            dim (int): The dimensionality of the input tensor (default is 256).
            fc_layer_dropout (float): Dropout probability applied after each fully connected layer (default is 0.0).
        """
        super(Regression, self).__init__()

        # Linear layer to reduce the last dimension from `dim` to 6
        self.dim_reduction_fc = nn.Linear(dim, 6)
        
        # Dropout layer with the specified probability
        self.dropout = nn.Dropout(fc_layer_dropout)
        
        # Fully connected layers for rotation prediction
        self.fc1_rot = nn.Linear(in_channels * 6, fc_hidden_layer_size)
        self.fc2_rot = nn.Linear(fc_hidden_layer_size, fc_hidden_layer_size // 2)
        self.fc3_rot = nn.Linear(fc_hidden_layer_size // 2, fc_hidden_layer_size // 8)
        
        # Fully connected layers for translation prediction
        self.fc1_tr = nn.Linear(in_channels * 6, fc_hidden_layer_size)
        self.fc2_tr = nn.Linear(fc_hidden_layer_size, fc_hidden_layer_size // 2)
        self.fc3_tr = nn.Linear(fc_hidden_layer_size // 2, fc_hidden_layer_size // 8)
        
        # Output layers for translation and rotation vectors
        self.translation_head = nn.Linear(fc_hidden_layer_size // 8, 3)
        self.rotation_head = nn.Linear(fc_hidden_layer_size // 8, 3)

        # GELU activation function
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the regression network to predict rotation and translation.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D], where B is the batch size,
                              C is the number of channels, and D is the dimension of the input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Rotation vector with shape [B, 3]
                - Translation vector with shape [B, 3]
        """
        B, C, D = x.shape

        # Change shape to [batch_size, dim, in_channels] for the FC layer
        x = x.permute(0, 2, 1)  # From [batch_size, in_channels, dim] to [batch_size, dim, in_channels]
        
        # Flatten the tensor to shape [batch_size * dim, in_channels]
        x = x.reshape(-1, D)
        
        # Apply dimension reduction FC layer
        x = self.dim_reduction_fc(x)  # Output shape: [batch_size * dim, 6]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Flatten the tensor for the next FC layers
        x = x.reshape(B, -1)  # Flatten to shape [batch_size, in_channels * 6]
        
        # Fully connected layers for rotation prediction
        x_rot = self.gelu(self.fc1_rot(x)) 
        x_rot = self.dropout(x_rot)
        x_rot = self.gelu(self.fc2_rot(x_rot))
        x_rot = self.dropout(x_rot)
        x_rot = self.gelu(self.fc3_rot(x_rot))
        x_rot = self.dropout(x_rot)
        
        # Fully connected layers for translation prediction
        x_tr = self.gelu(self.fc1_tr(x))
        x_tr = self.dropout(x_tr)
        x_tr = self.gelu(self.fc2_tr(x_tr))
        x_tr = self.dropout(x_tr)
        x_tr = self.gelu(self.fc3_tr(x_tr))
        x_tr = self.dropout(x_tr)
        
        # Separate heads for translation and rotation outputs
        x_rot = self.rotation_head(x_rot)  # Rotation output shape: [batch_size, 3]
        x_tr = self.translation_head(x_tr)  # Translation output shape: [batch_size, 3]
        
        return x_rot, x_tr



class RegressionV1(nn.Module):
    def __init__(self, in_channels:int=512, fc_hidden_layer_size:int=128, dim:int=256, fc_layer_dropout:float=0.)->None:
        super(RegressionV1, self).__init__()
        
        # Linear layer to reduce the last dimension from `dim` to 6
        self.dim_reduction_fc = nn.Linear(dim, 6)
        
        # Dropout layer with the specified probability
        self.dropout = nn.Dropout(fc_layer_dropout)
        
        # Fully connected layers for rotation prediction
        self.fc1_rot = nn.Linear(in_channels * 6, fc_hidden_layer_size)
        self.fc2_rot = nn.Linear(fc_hidden_layer_size, fc_hidden_layer_size // 2)
        #self.fc3_rot = nn.Linear(fc_hidden_layer_size//2, fc_hidden_layer_size // 8)
        
        # Fully connected layers for translation prediction
        self.fc1_tr = nn.Linear(in_channels * 6, fc_hidden_layer_size)
        self.fc2_tr = nn.Linear(fc_hidden_layer_size, fc_hidden_layer_size // 2)
        #self.fc3_tr = nn.Linear(fc_hidden_layer_size//2, fc_hidden_layer_size // 8)
        
        # Output layers for translation and rotation vectors
        self.translation_head = nn.Linear(fc_hidden_layer_size // 2, 3)
        self.rotation_head = nn.Linear(fc_hidden_layer_size // 2, 3)

        self.gelu = nn.GELU()
        
    def forward(self, x:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:

        B, C, D = x.shape

        # Change shape to [batch_size, dim, in_channels] for the FC layer
        x = x.permute(0, 2, 1)  # From [batch_size, in_channels, dim] to [batch_size, dim, in_channels]
        
        # Flatten the tensor to shape [batch_size * dim, in_channels]
        x = x.reshape(-1, D)
        
        # Apply dimension reduction FC layer
        x = self.dim_reduction_fc(x)  # Output shape: [batch_size * dim, 6]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Flatten the tensor for the next FC layers
        x = x.reshape(B, -1)  # Flatten to shape [batch_size, in_channels * 6]
        
        # Fully connected layers for rotation prediction
        x_rot =  self.gelu(self.fc1_rot(x)) 
        x_rot = self.dropout(x_rot)
        x_rot = self.gelu(self.fc2_rot(x_rot))
        x_rot = self.dropout(x_rot)

        # Fully connected layers for translation prediction
        x_tr = self.gelu(self.fc1_tr(x))
        x_tr = self.dropout(x_tr)
        x_tr = self.gelu(self.fc2_tr(x_tr))
        x_tr = self.dropout(x_tr)
       
        # Separate heads for translation and rotation outputs
        x_rot = self.rotation_head(x_rot)
        x_tr = self.translation_head(x_tr)
        
        return x_rot, x_tr