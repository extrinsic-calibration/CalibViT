import torch 
from torch import nn
from typing import Tuple, Optional

# Debugging class for Adaptive Max Pooling
class MyAdaptiveMaxPool2d(nn.Module):
    """
    A custom implementation of Adaptive Max Pooling for debugging purposes. 
    This class replaces AdaptiveMaxPool2d and is useful when exporting to ONNX.

    Args:
        sz (tuple, optional): Target output size for the pooling. Default is None.

    Inputs:
        x (torch.Tensor): Input tensor of shape [B, C, H, W], where
            B: Batch size,
            C: Number of channels,
            H: Height,
            W: Width.

    Outputs:
        torch.Tensor: Max-pooled tensor with reduced spatial dimensions.
    """
    def __init__(self, sz=None):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the custom adaptive max pooling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after max pooling.
        """
        inp_size = x.size()  # Input tensor dimensions
        print(inp_size)  # Debugging: Print input size
        # Perform max pooling with kernel size equal to the input dimensions
        return nn.functional.max_pool2d(
            input=x,
            kernel_size=(inp_size[2], inp_size[3])
        )


# Downsampling block to resize backbone features
class DownsampleBlock(nn.Module):
    """
    A downsampling module to resize backbone output for the patch embedding module. 
    Adapts behavior based on the dataset being used (e.g., nuScenes, KITTI).

    Args:
        in_channels (int): Number of input feature channels. Default is 512.
        out_channels (int): Number of output feature channels. Default is 512.
        patch_size_h (int): Target patch height. Default is 16.
        patch_size_w (int): Target patch width. Default is 16.
        dataset (str): Dataset type ("nuscenes" or "kitti"). Default is "nuscenes".

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies downsampling to the input tensor.

    Inputs:
        x (torch.Tensor): Input tensor of shape [B, C, H, W].

    Outputs:
        torch.Tensor: Downsampled tensor of shape [B, out_channels, patch_size_h, patch_size_w].
    """
    def __init__(
        self,
        in_channels=512,
        out_channels=512,
        patch_size_h=16,
        patch_size_w=16,
        dataset="nuscenes"
    ):
        super(DownsampleBlock, self).__init__()
        self.dataset = dataset

        if self.dataset == "nuscenes":
            # Initial MaxPool layer for downsampling
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # BasicBlock layers for further feature processing
            self.basicblock = nn.Sequential(
                BasicBlock(in_channels, out_channels, stride=2, dilation=1, downsample=nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(out_channels),
                )),
                BasicBlock(out_channels, out_channels, stride=1),
            )

            # Adaptive pooling to resize features to patch size
            self.maxpool2 = nn.AdaptiveMaxPool2d((patch_size_h, patch_size_w))
            # Uncomment for ONNX model export
            # self.maxpool2 = MyAdaptiveMaxPool2d()

        elif self.dataset == "kitti":
            # Simple AdaptiveMaxPool for KITTI
            self.maxpool3 = nn.AdaptiveMaxPool2d((patch_size_h, patch_size_w))

        # Initialize weights for all layers
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for downsampling.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Downsampled tensor with shape [B, out_channels, patch_size_h, patch_size_w].
        """
        if self.dataset == "nuscenes":
            # Complex downsampling pipeline for nuScenes dataset
            x = self.maxpool1(x)
            x = self.basicblock(x)
            x = self.maxpool2(x)
        elif self.dataset == "kitti":
            # Simple downsampling for KITTI dataset
            x = self.maxpool3(x)

        return x  # Output tensor of shape [B, out_channels, patch_size_h, patch_size_w]


# Initialize weights for the layers
def initialize_weights(m):
    """
    Initializes weights of Conv2D and Linear layers using Xavier initialization.
    Also initializes BatchNorm2D and LayerNorm layers with constant values.

    Args:
        m (nn.Module): Module to initialize.

    Effects:
        - Conv2D and Linear layers: Xavier uniform initialization.
        - BatchNorm2D and LayerNorm: Weights set to 1 and biases set to 0.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Initialize biases to 0
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)  # Set weights to 1
        nn.init.constant_(m.bias, 0)  # Set biases to 0

        


# Backbone for Feature Extraction
class FeatureExtraction(nn.Module):
    """
    Feature extraction module that uses ResNet18-based backbones for RGB and depth inputs.

    Args:
        planes (int): The number of output planes in the first ResNet layer. 
                      The number of output channels will scale accordingly. Default is 32.

    Inputs:
        rgb (torch.Tensor): Input RGB image of shape [B, 3, H, W].
        depth (torch.Tensor): Input depth map of shape [B, 1, H, W].

    Outputs:
        Tuple[torch.Tensor, torch.Tensor]: Extracted features for RGB and depth.
            - Both tensors are of shape [B, 512, H_out, W_out], where
    """
    def __init__(self, planes: int = 32) -> None:
        super(FeatureExtraction, self).__init__()
        # ResNet backbone for RGB input
        self.rgb_resnet = resnet18(inplanes=3, planes=planes)  # If planes = 32, outplanes = 512

        # ResNet backbone for depth input
        self.depth_resnet = resnet18(inplanes=3, planes=planes // 2)  # Halve the planes for depth

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for feature extraction.

        Args:
            rgb (torch.Tensor): RGB input tensor of shape [B, 3, H, W].
            depth (torch.Tensor): Depth input tensor of shape [B, 1, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Extracted RGB features of shape [B, 512, H_out, W_out].
                - Extracted depth features of shape [B, 512, H_out, W_out].
        """
        # Clone depth tensor to avoid modifying the original input
        x1, x2 = rgb, depth.clone()

        # Process RGB input through the RGB ResNet backbone
        x1 = self.rgb_resnet(x1)[-1]  # Extract features from the last layer

        # Process Depth input through the Depth ResNet backbone
        x2 = self.depth_resnet(x2)[-1]  # Extract features from the last layer

        
        return x1, x2  # Return RGB and depth features


# Basic Block for ResNet
class BasicBlock(nn.Module):
    """
    Basic building block for ResNet architecture.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the first convolution. Default is 1.
        dilation (int, optional): Dilation rate for convolutions. Default is 1.
        downsample (nn.Module, optional): Downsampling layer to match dimensions. Default is None.

    Input:
        x (torch.Tensor): Input tensor of shape [B, inplanes, H, W].

    Output:
        torch.Tensor: Output tensor of shape [B, planes, H_out, W_out].
    """
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 dilation: int = 1, downsample: nn.Module = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [B, inplanes, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [B, planes, H_out, W_out].
        """
        residual = x  # Save the input as residual for the skip connection

        # First convolution + batch norm + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution + batch norm
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling if provided
        if self.downsample is not None:
            residual = self.downsample(x)

        # Add the residual connection and apply ReLU
        out += residual
        out = self.relu(out)

        return out


# Linear Projection Layer
class LinearProjection(nn.Module):
    def __init__(self, resnet_output_shape, num_patches, embed_dim):
        super(LinearProjection, self).__init__()
        H, W = resnet_output_shape
        # Linear layer for projecting 96x56 to embed_dim
        self.projection = nn.Linear(H * W, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Flatten the spatial dimensions
        x = x.view(B, C, H * W)  # (B, C, H*W)
       
        # Project to the embedding dimension
        x = self.projection(x)  # (B, C, embed_dim)
    
        return x

class resnet18(nn.Module):
    """
    Custom implementation of ResNet-18 for feature extraction.

    Args:
        inplanes (int, optional): Number of input channels. Default is 3 (e.g., RGB images).
        planes (int, optional): Number of output channels in the first convolution. Default is 64.

    Input:
        x (torch.Tensor): Input tensor of shape [B, inplanes, H, W].

    Output:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Outputs from each block:
            - out1: Output from layer1, shape [B, planes, H/4, W/4].
            - out2: Output from layer2, shape [B, planes*2, H/8, W/8].
            - out3: Output from layer3, shape [B, planes*4, H/8, W/8].
            - out4: Output from layer4, shape [B, planes*8, H/8, W/8].
    """
    def __init__(self, inplanes: int = 3, planes: int = 64) -> None:
        super(resnet18, self).__init__()

        # Initial convolutional stem with downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(inplanes, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        # Max-pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block groups
        self.layer1 = nn.Sequential(
            BasicBlock(planes, planes, stride=1, dilation=1),
            BasicBlock(planes, planes, stride=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(planes, planes * 2, stride=2, dilation=1, downsample=nn.Sequential(
                nn.Conv2d(planes, planes * 2, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * 2),
            )),
            BasicBlock(planes * 2, planes * 2, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(planes * 2, planes * 4, stride=1, downsample=nn.Sequential(
                nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * 4),
            )),
            BasicBlock(planes * 4, planes * 4, stride=1, dilation=2),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(planes * 4, planes * 8, stride=1, dilation=2, downsample=nn.Sequential(
                nn.Conv2d(planes * 4, planes * 8, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * 8),
            )),
            BasicBlock(planes * 8, planes * 8, stride=1, dilation=4),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [B, inplanes, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - out1: Output from layer1.
                - out2: Output from layer2.
                - out3: Output from layer3.
                - out4: Output from layer4.
        """
        # Pass through the stem and pooling layer
        out = self.stem(x)
        out = self.maxpool(out)

        # Pass through residual blocks
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return out1, out2, out3, out4


class ConvStem(nn.Module):
    """
    Convolutional Stem module for feature extraction and patch embedding.

    This module extracts features using a sequence of convolutional blocks, 
    then downsamples and projects them to a lower-dimensional embedding space. 
    Inspired by the SalsaNext architecture.

    Args:
        in_channels (int): Number of input channels. Default is 5.
        base_channels (int): Number of base channels for the initial convolutional block. Default is 32.
        img_size (Tuple[int, int]): Input image size (H, W). Default is (32, 384).
        patch_stride (Tuple[int, int]): Stride for patch extraction. Default is (2, 8).
        embed_dim (int): Dimensionality of the embedding space. Default is 384.
        flatten (bool): Whether to flatten and rearrange the output tensor into (B, N, C). Default is True.
        hidden_dim (Optional[int]): Number of channels in the intermediate layers. 
                                    If None, defaults to 2 * base_channels.

    Input:
        x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].

    Output:
        Tuple[torch.Tensor, torch.Tensor]:
            - x: Patch-embedded tensor of shape [B, num_patches, embed_dim] (if flatten=True) or 
                 [B, embed_dim, H_out, W_out] (if flatten=False).
            - x_base: Intermediate feature map tensor of shape [B, hidden_dim, H, W].
    """
    def __init__(self,
                 in_channels: int = 5,
                 base_channels: int = 32,
                 img_size: Tuple[int, int] = (32, 384),
                 patch_stride: Tuple[int, int] = (2, 8),
                 embed_dim: int = 384,
                 flatten: bool = True,
                 hidden_dim: Optional[int] = None) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 2 * base_channels

        self.base_channels = base_channels
        self.dropout_ratio = 0.2

        # Convolutional block for feature extraction
        self.conv_block = nn.Sequential(
            ResContextBlock(in_channels, base_channels),
            ResContextBlock(base_channels, base_channels),
            ResContextBlock(base_channels, base_channels),
            ResBlock(base_channels, hidden_dim, self.dropout_ratio, pooling=False, drop_out=False)
        )

        # Downsampling and projection block
        assert patch_stride[0] % 2 == 0
        assert patch_stride[1] % 2 == 0
        kernel_size = (patch_stride[0] + 1, patch_stride[1] + 1)
        padding = (patch_stride[0] // 2, patch_stride[1] // 2)
        self.proj_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=patch_stride, padding=padding),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        )

        self.patch_stride = patch_stride
        self.patch_size = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

    def get_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the grid size based on input dimensions.

        Args:
            H (int): Input height.
            W (int): Input width.

        Returns:
            Tuple[int, int]: Grid size (H_out, W_out) after patch extraction.
        """
        return get_grid_size_2d(H, W, self.patch_size, self.patch_stride)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the module.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x: Patch-embedded tensor of shape [B, num_patches, embed_dim] (if flatten=True) or 
                     [B, embed_dim, H_out, W_out] (if flatten=False).
                - x_base: Intermediate feature map tensor of shape [B, hidden_dim, H, W].
        """
        B, C, H, W = x.shape  # Input dimensions: [B, in_channels, H, W]

        # Feature extraction
        x_base = self.conv_block(x)  # [B, hidden_dim, H, W]

        # Downsampling and projection
        x = self.proj_block(x_base)  # [B, embed_dim, H_out, W_out]

        # Optionally flatten the output for patch embeddings
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, embed_dim, H_out*W_out] -> [B, num_patches, embed_dim]

        return x, x_base


class ResContextBlock(nn.Module):
    """
    Residual Context Block for feature extraction, as used in SalsaNext architecture by Tiago Cortinhal et al.
    
    This block consists of a series of convolutions and activations with a residual shortcut connection.
    The block applies a sequence of 1x1, 3x3, and 3x3 convolutions with dilation, followed by LeakyReLU activations 
    and Batch Normalization, to extract context-aware features.

    Args:
        in_filters (int): Number of input channels to the block.
        out_filters (int): Number of output channels for the block.

    Output:
        torch.Tensor: Output tensor of shape [B, out_filters, H, W], where out_filters is the number of output channels.
    """

    def __init__(self, in_filters: int, out_filters: int) -> None:
        """
        Initializes the ResContextBlock with specified input and output channels.

        Args:
            in_filters (int): Number of input channels to the block.
            out_filters (int): Number of output channels for the block.
        """
        super(ResContextBlock, self).__init__()

        # Convolutional layers with activations and batch normalization
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResContextBlock.

        Applies a series of convolutions, activations, and batch normalization layers with a residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_filters, H, W], where B is the batch size, 
                              in_filters is the number of input channels, and H, W are the spatial dimensions.

        Returns:
            torch.Tensor: Output tensor of shape [B, out_filters, H, W], where out_filters is the number of output channels.
        """
        # Shortcut path (1x1 convolution)
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        # Main residual path
        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        # Adding the shortcut connection to the result
        output = shortcut + resA2

        return output

class ResBlock(nn.Module):
    """
    A residual block used in SalsaNext architecture for feature extraction with multiple convolutions 
    and activations, followed by a residual connection. The block can include pooling and dropout for regularization.

    Args:
        in_filters (int): Number of input channels to the block.
        out_filters (int): Number of output channels for the block.
        dropout_rate (float): Dropout rate for regularization.
        kernel_size (Tuple[int, int]): The kernel size for pooling (default is (3, 3)).
        stride (int): Stride of the convolutions (default is 1).
        pooling (bool): Whether to apply pooling (default is True).
        drop_out (bool): Whether to apply dropout (default is True).

    """
    
    def __init__(self, in_filters: int, out_filters: int, dropout_rate: float, kernel_size: Tuple[int, int] = (3, 3), 
                 stride: int = 1, pooling: bool = True, drop_out: bool = True) -> None:
        """
        Initializes the ResBlock with specified parameters.

        Args:
            in_filters (int): Number of input channels.
            out_filters (int): Number of output channels.
            dropout_rate (float): Dropout rate.
            kernel_size (Tuple[int, int]): Pooling kernel size.
            stride (int): Stride for convolutions.
            pooling (bool): Whether to apply pooling.
            drop_out (bool): Whether to apply dropout.
        """
        super(ResBlock, self).__init__()

        self.pooling = pooling
        self.drop_out = drop_out

        # Define the layers
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        # Dropout and pooling layers
        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] or torch.Tensor:
        """
        Forward pass through the ResBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_filters, H, W], where B is the batch size,
                              in_filters is the number of input channels, and H, W are the spatial dimensions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] or torch.Tensor:
                - If pooling is enabled, returns a tuple:
                    - The pooled output tensor (resB) with shape [B, out_filters, H', W']
                    - The non-pooled output (resA) with shape [B, out_filters, H', W']
                - If pooling is disabled, returns only the non-pooled tensor (resB).
        """
        # Shortcut connection (1x1 convolution)
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        # First residual path
        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        # Second residual path with dilation
        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        # Third residual path with dilation
        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        # Concatenate residual paths and apply final 1x1 convolution
        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)

        # Add shortcut to the final result (residual connection)
        resA = shortcut + resA

        # Apply dropout and pooling if enabled
        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)
            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


def get_grid_size_2d(H: int, W: int, patch_size: int, patch_stride: Optional[int]) -> Tuple[int, int]:
    """
    Calculates the grid size (height and width) for 2D patches given the image dimensions, 
    patch size, and stride.

    Args:
        H (int): Height of the input image.
        W (int): Width of the input image.
        patch_size (int): Size of each patch (assumed square).
        patch_stride (Optional[int]): Stride for patching. If None, stride is equal to patch size.

    Returns:
        Tuple[int, int]: Grid height and width (number of patches in each dimension).
    """
    if isinstance(patch_size, int):
        PS_H = PS_W = patch_size
    else:
        PS_H, PS_W = patch_size

    if patch_stride is not None:
        if isinstance(patch_stride, int):
            patch_stride = (patch_stride, patch_stride)
        H_stride, W_stride = patch_stride
    else:
        H_stride = PS_H
        W_stride = PS_W

    grid_H = get_grid_size_1d(H, PS_H, H_stride)
    grid_W = get_grid_size_1d(W, PS_W, W_stride)
    return grid_H, grid_W


def get_grid_size_1d(length: int, patch_size: int, stride: int) -> int:
    """
    Calculates the number of patches in one dimension (height or width).

    Args:
        length (int): Length of the image dimension (height or width).
        patch_size (int): Size of the patch.
        stride (int): Stride for patching.

    Returns:
        int: Number of patches in the given dimension.
    """
    assert patch_size % stride == 0
    assert length % patch_size == 0
    return (length - patch_size) // stride + 1
