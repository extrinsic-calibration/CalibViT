import torch
from torch import nn
from typing import Optional

def initialize_weights(m: nn.Module) -> None:
    """
    Initialize weights for different layers in the network using Xavier initialization.
    - Convolutional and Linear layers are initialized with Xavier uniform distribution.
    - BatchNorm2d and LayerNorm layers are initialized with ones for weights and zeros for biases.

    Args:
        m (nn.Module): The layer to initialize weights for.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Xavier uniform initialization for weights
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # Initialize bias with zeros
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        # BatchNorm and LayerNorm weights are initialized to 1, and biases to 0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class FeaturePatchEmbed(nn.Module):
    """
    A module that transforms the input image into a series of flattened patches, each of which
    corresponds to a different feature channel. The module adds positional embeddings to each patch.

    Args:
        patch_size_h (int): Height of each patch (default is 16).
        patch_size_w (int): Width of each patch (default is 16).
        in_chans (int): Number of input channels (default is 512).
    """
    
    def __init__(self, patch_size_h: int = 16, patch_size_w: int = 16, in_chans: int = 512) -> None:
        """
        Initializes the FeaturePatchEmbed module.

        Args:
            patch_size_h (int): Height of each patch (default is 16).
            patch_size_w (int): Width of each patch (default is 16).
            in_chans (int): Number of input channels (default is 512).
        """
        super().__init__()
        
        # Number of patches is equal to the number of input channels
        num_patches = in_chans  
        # Embed dimension is the product of the patch height and width
        embed_dim = patch_size_h * patch_size_w 
        
        self.img_height = patch_size_h
        self.img_width = patch_size_w
        self.in_chans = in_chans
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Flatten layer to convert image into patches (each patch is a vector)
        self.flatten_layer = nn.Flatten(start_dim=2, end_dim=3)
        
        # Positional embeddings to retain the spatial information of patches
        self.position_embeddings = nn.Parameter(torch.rand((1, num_patches, embed_dim), requires_grad=True))
        
        # Apply weight initialization
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that flattens the input image into patches and adds positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_chans, H, W], where B is the batch size,
                              in_chans is the number of input channels, and H, W are the height and width of the image.

        Returns:
            torch.Tensor: The output tensor with shape [B, num_patches, embed_dim].
        """
        # Flatten the input image to obtain patches (flatten the H and W dimensions of each patch)
        x = self.flatten_layer(x)
        
        # Add the position embeddings to each patch
        x = x + self.position_embeddings
        
        return x  # Shape: [B, num_patches, embed_dim]
