import torch 
from torch import nn
from timm.models.layers import trunc_normal_, DropPath

from .swin_transformer import SwinTransformer
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# weights for ViT
def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# Cross-Attention module definition
class CrossAttention(nn.Module):
    """
    Implements a multi-head cross-attention mechanism where the input tensor `x` is used as the key-value pair, 
    and the `query` tensor is used as the query in the attention mechanism.
    """
    def __init__(self, dim: int, heads: int, dropout: float):
        """
        Initializes the CrossAttention module with the given dimensions, number of attention heads, and dropout.

        Args:
            dim (int): The input dimension of the features.
            heads (int): The number of attention heads.
            dropout (float): The dropout rate for regularization.
        """
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5  # Scaling factor for attention logits
        self.attn = None
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.kv = nn.Linear(dim, dim * 2)
        self.q = nn.Linear(dim, dim)

    @property
    def unwrapped(self):
        """
        Returns the unwrapped CrossAttention module.
        """
        return self

    def forward(self, x: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Forward pass for the CrossAttention module. Computes the attention weights and outputs the attended features.

        Args:
            x (torch.Tensor): The key-value tensor, shape (B, N, C) where B is batch size, N is sequence length, and C is feature dimension.
            query (torch.Tensor): The query tensor, shape (B, N, C).
            mask (torch.Tensor, optional): The attention mask tensor to avoid attending to certain positions, shape (B, N, N).

        Returns:
            tuple: The attended feature tensor (B, N, C) and attention weights (B, heads, N, N).
        """
        B, N, C = x.shape

        q = self.q(query).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))

        return x, attn


# FeedForward module definition
class FeedForward(nn.Module):
    """
    Implements a simple feedforward neural network with two fully connected layers and GELU activation in between.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float, out_dim: int = None):
        """
        Initializes the FeedForward module with the given input dimension, hidden dimension, dropout, and output dimension.

        Args:
            dim (int): The input dimension of the features.
            hidden_dim (int): The hidden dimension of the feedforward network.
            dropout (float): The dropout rate for regularization.
            out_dim (int, optional): The output dimension. If None, the output dimension will be the same as the input dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        """
        Returns the unwrapped FeedForward module.
        """
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FeedForward module. Applies two fully connected layers with GELU activation and dropout.

        Args:
            x (torch.Tensor): The input tensor, shape (B, N, C).

        Returns:
            torch.Tensor: The output tensor, shape (B, N, C) after applying the feedforward network.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# LayerScale module definition
class LayerScale(nn.Module):
    """
    Implements a learnable scaling factor (gamma) for the input tensor, which is element-wise multiplied.
    """
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        """
        Initializes the LayerScale module with the given dimensions, initial gamma values, and inplace option.

        Args:
            dim (int): The dimension of the input tensor to scale.
            init_values (float): The initial values for the scaling factor (gamma).
            inplace (bool): Whether to apply the scaling in-place (modifying the input tensor).
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LayerScale module. Scales the input tensor by the learnable gamma factor.

        Args:
            x (torch.Tensor): The input tensor, shape (B, N, C).

        Returns:
            torch.Tensor: The scaled tensor, shape (B, N, C).
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# CrossAttentionBlock definition
class CrossAttentionBlock(nn.Module):
    """
    Implements a CrossAttention block consisting of multi-head cross-attention, layer scaling, and feedforward network.
    This block is useful for attention-based models.
    """
    def __init__(self, dim: int = 256, heads: int = 8, mlp_dim: int = 1024, dropout: float = 0., drop_path: float = 0., 
                 reduction_dim: int = None, init_values: float = None):
        """
        Initializes the CrossAttentionBlock with the given parameters.

        Args:
            dim (int): The input/output dimension of the features.
            heads (int): The number of attention heads.
            mlp_dim (int): The hidden dimension of the feedforward network.
            dropout (float): The dropout rate for regularization.
            drop_path (float): The drop path rate for regularization.
            reduction_dim (int, optional): If specified, applies dimensionality reduction.
            init_values (float, optional): The initial scaling values for layer scaling.
        """
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.reduction_dim = reduction_dim
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, heads, dropout)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        if self.reduction_dim:
            self.dim_reduction = nn.Linear(dim, reduction_dim)
            self.dim_reduction_q = nn.Linear(dim, reduction_dim)
            self.norm3 = nn.LayerNorm(reduction_dim)
            self.normquery = nn.LayerNorm(reduction_dim)
        else:
            self.dim_reduction = None

    def forward(self, x: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False) -> tuple:
        """
        Forward pass for the CrossAttentionBlock. Applies cross-attention, layer scaling, and feedforward network.

        Args:
            x (torch.Tensor): The key-value tensor, shape (B, N, C).
            query (torch.Tensor): The query tensor, shape (B, N, C).
            mask (torch.Tensor, optional): The attention mask tensor to avoid attending to certain positions, shape (B, N, N).
            return_attention (bool, optional): Whether to return the attention weights.

        Returns:
            tuple: The processed tensor (B, N, C) and the query tensor (B, N, C) after dimensionality reduction (if applied).
        """
        B, N, C = x.shape
        y, attn = self.attn(self.norm1(x), self.norm1(query),  mask)
        
        if return_attention:
            return attn
        
        y = self.ls1(y)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if self.dim_reduction:
            x = self.dim_reduction(x)
            q = self.dim_reduction_q(query)
            q = self.normquery(q)
        else:  
            q = query 
        
        return x, q


class VisionTransformer(nn.Module):
    """
    A Vision Transformer (ViT) model that applies a series of Cross-Attention blocks on the input tensor.
    This model can be used for various vision tasks by learning global features via self-attention mechanisms.

    The model consists of multiple Cross-Attention blocks, followed by layer normalization and an optional return of attention weights.
    """
    def __init__(
        self,
        dim: int = 256,
        heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        n_crosslayers: int = 6
    ):
        """
        Initializes the VisionTransformer model.

        Args:
            dim (int): Dimension of the input and output embeddings (default: 256).
            heads (int): Number of attention heads in each Cross-Attention block (default: 8).
            mlp_dim (int): Dimension of the MLP in the transformer blocks (default: 1024).
            dropout (float): Dropout probability for attention and MLP layers (default: 0.0).
            drop_path_rate (float): Drop path rate for regularization in the transformer blocks (default: 0.0).
            n_crosslayers (int): Number of Cross-Attention layers in the Vision Transformer (default: 6).
        """
        super().__init__()

        # Compute drop path rates for each Cross-Attention block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_crosslayers)]
        
        # Create a list of Cross-Attention blocks with the given parameters
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(dim, heads, mlp_dim, dropout, dpr[i], init_values=None, reduction_dim=None) for i in range(n_crosslayers)]
        )
        
        # Layer normalization applied at the end of the VisionTransformer
        self.norm = nn.LayerNorm(dim)

        # List to store attention weights if needed
        self.attn_weights = []

    def forward(self, x: torch.Tensor, query: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass through the VisionTransformer model.

        Args:
            x (torch.Tensor): The input tensor, shape [B, N, D], where B is the batch size, N is the number of tokens (sequence length), and D is the feature dimension.
            query (torch.Tensor): The query tensor used for cross-attention, shape [B, N, D].
            return_attn (bool, optional): Whether to return attention weights (default: False).

        Returns:
            torch.Tensor: The output tensor, shape [B, N, D], after passing through all Cross-Attention blocks and layer normalization.
            Optional[List]: A list of attention weights for each Cross-Attention block, returned only if `return_attn` is True.
        """
        # Iterate through each Cross-Attention block and apply them to the input tensor
        for blk in self.blocks:
            if return_attn:
                x, attn = blk(x=x, query=query, return_attention=return_attn)
                self.attn_weights.append(attn)
            else:
                x, q = blk(x, query)

        # Apply layer normalization to the final output
        x = self.norm(x)

        # Return the output tensor and attention weights if requested
        if return_attn:
            return x, self.attn_weights
        
        return x  # Output tensor shape: [B, N, D]


class SwinFeatureExtraction(nn.Module):
    """
    A feature extraction module using Swin Transformer for both RGB and depth images.
    
    This module processes both RGB and depth images through separate Swin Transformer backbones.
    The images are resized to 224x224 before passing through the respective backbones to extract features.
    
    Attributes:
        maxpoolrbg (nn.AdaptiveMaxPool2d): A pooling layer to resize RGB images to 224x224.
        maxpooldepth (nn.AdaptiveMaxPool2d): A pooling layer to resize depth images to 224x224.
        rgb_swin (nn.Sequential): A sequence of layers from the Swin Transformer for RGB images.
        depth_swin (nn.Sequential): A sequence of layers from the Swin Transformer for depth images.
    """

    def __init__(self, backbone_pretrained: bool = False, planes: int = 32):
        """
        Initializes the SwinFeatureExtraction module.

        Args:
            backbone_pretrained (bool): Whether to use a pre-trained Swin Transformer backbone. 
                                         (Currently not used in this implementation.)
            planes (int): The number of input channels for the Swin Transformer. 
                          For RGB images, this is 3, and for depth images, it is 1.
        """
        super(SwinFeatureExtraction, self).__init__()

        # Resize input images to 224x224 using max pooling
        self.maxpoolrbg = nn.AdaptiveMaxPool2d((224, 224))
        self.maxpooldepth = nn.AdaptiveMaxPool2d((224, 224))

        # Initialize Swin Transformer models for RGB and depth images
        self.rgb_swin = SwinTransformer(img_size=224, in_chans=planes)
        # Disable parts of the Swin Transformer that are not needed
        self.rgb_swin.avgpool = nn.Identity()
        self.rgb_swin.norm = nn.Identity()
        self.rgb_swin.head = nn.Identity()
        
        self.depth_swin = SwinTransformer(img_size=224, in_chans=planes)
        # Disable parts of the Swin Transformer for depth images
        self.depth_swin.avgpool = nn.Identity()
        self.depth_swin.norm = nn.Identity()
        self.depth_swin.head = nn.Identity()

        # Apply weight initialization (optional method if used in your implementation)
        self.apply(self.initialize_weights)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the feature extraction model.

        Args:
            rgb (torch.Tensor): Input RGB images with shape [B, 3, H, W] where 
                                B is batch size, 3 is the number of channels (RGB), 
                                H is the height (e.g., 900 or 376), W is the width (e.g., 1600 or 1241).
            depth (torch.Tensor): Input depth images with shape [B, 1, H, W] where 
                                  B is batch size, 1 is the number of channels (depth), 
                                  H is the height, W is the width.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Features extracted from RGB images with shape [B, C, H', W'] where 
                                C is the number of channels after processing, H' and W' are the reduced dimensions.
                - torch.Tensor: Features extracted from depth images with shape [B, C, H', W'] where 
                                C is the number of channels after processing, H' and W' are the reduced dimensions.
        """
        # Resize the input RGB and depth images
        x1 = self.maxpoolrbg(rgb)  # Resize RGB images to 224x224
        x2 = self.maxpooldepth(depth)  # Resize depth images to 224x224
        
        # Extract features from the resized RGB and depth images using their respective Swin Transformers
        x1 = self.rgb_swin(x1)  # Features from RGB images
        x2 = self.depth_swin(x2)  # Features from depth images

        # Return the extracted features
        return x1, x2


class Transformer(nn.Module):
    """
    Transformer model with multiple CrossAttentionBlocks.
    
    This model consists of a sequence of cross-attention layers, with optional dimensional reduction
    after every two layers. The cross-attention blocks are designed to process input data in a multi-head attention 
    mechanism, with an MLP layer for additional transformation.
    
    Attributes:
        blocks (nn.ModuleList): List of CrossAttentionBlock layers.
        norm (nn.LayerNorm): Layer normalization applied to the final output.
    """

    def __init__(
        self,
        dim: int = 256,                    # Dimension of the input and output embeddings
        heads: List[int] = [6, 8, 12, 12, 24, 24],  # List of attention heads per layer
        mlp_dim: int = 1024,               # Dimension of the MLP inside the transformer blocks
        dropout: float = 0.0,              # Dropout rate for attention and MLP layers
        drop_path_rate: float = 0.0,       # Drop path rate for transformer layers
        n_crosslayers: int = 6,            # Number of cross-attention layers
        reduction_dims: Optional[List[int]] = None  # List of dimensions for optional reduction [256, 128]
    ):
        """
        Initializes the Transformer model with multiple CrossAttentionBlock layers.

        Args:
            dim (int): The dimension of input/output embeddings.
            heads (List[int]): List of the number of attention heads for each cross-attention layer.
            mlp_dim (int): The dimension of the MLP inside each transformer block.
            dropout (float): Dropout probability for attention and MLP layers.
            drop_path_rate (float): Drop path rate applied to each cross-attention layer.
            n_crosslayers (int): The number of cross-attention layers to use.
            reduction_dims (List[int], optional): Dimensions to reduce to after each cross-attention block. 
                                                   Applied after every second block if specified.
        """
        super(Transformer, self).__init__()

        # Compute drop path rates for each cross-attention layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_crosslayers)]
        
        # List of transformer blocks (CrossAttentionBlock layers)
        self.blocks = nn.ModuleList()
        
        for i in range(n_crosslayers):
            reduction_dim = None
            
            if reduction_dims is not None:
                if i > 0 and i % 2 == 1:  # Apply reduction after every two layers
                    if len(reduction_dims) > 0:
                        reduction_dim = reduction_dims.pop(0)
                    
                    # Add CrossAttentionBlock with reduction
                    self.blocks.append(
                        CrossAttentionBlock(dim, heads[i], dim * 4, dropout, dpr[i], reduction_dim=reduction_dim)
                    )
                    # Update dim to the reduced dimension for subsequent layers
                    dim = reduction_dim
                else:
                    # Add CrossAttentionBlock without reduction
                    self.blocks.append(
                        CrossAttentionBlock(dim, heads[i], dim * 4, dropout, dpr[i], reduction_dim=None)
                    )
            else:
                # Add CrossAttentionBlock without reduction
                self.blocks.append(
                    CrossAttentionBlock(dim, heads[i], dim * 4, dropout, dpr[i], reduction_dim=None)
                )
        
        # Layer normalization to normalize the final output
        self.norm = nn.LayerNorm(dim)

        # Apply weight initialization (method should be defined elsewhere in the code)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D], where B is the batch size, 
                              N is the number of tokens (e.g., sequence length), 
                              and D is the dimension of the input.
            query (torch.Tensor): Query tensor for cross-attention, typically of shape [B, N, D].

        Returns:
            torch.Tensor: Output tensor after processing through the transformer blocks, 
                          with shape [B, N, D].
        """
        # Pass input through each CrossAttentionBlock in sequence
        for blk in self.blocks:
            x, query = blk(x, query)

        # Normalize the final output using LayerNorm
        x = self.norm(x)

        return x  # Output shape: [B, N, D]
