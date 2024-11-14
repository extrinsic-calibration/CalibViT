import torch 
from torch import nn
from typing import Tuple
import timm

from .blocks import FeatureExtraction, DownsampleBlock, LinearProjection
from .blocks import ConvStem
from .vision_transformer import VisionTransformer, SwinFeatureExtraction, Transformer
from .regression import Regression 
from .create_tokens import FeaturePatchEmbed
import math

from config import ModelConfig


class CalibViTV1(nn.Module):
    """
    CalibViTV1 is a neural network model that integrates RGB and depth image features for regression tasks. 
    It extracts features from RGB and depth inputs, processes them through a Vision Transformer (ViT) with 
    cross-attention mechanisms, and performs regression to predict calibration parameters (rotation and translation).

    Attributes:
        feature_extractor (FeatureExtraction): Feature extraction module for RGB and depth images.
        resnet_to_vit (LinearProjection): Linear projection layer to transform ResNet output to ViT input.
        depth_resnet_to_vit (LinearProjection): Linear projection layer to transform depth ResNet output to ViT input.
        positional_embeddings (torch.Tensor): Sinusoidal positional embeddings for the transformer.
        vit (VisionTransformer): Vision transformer block for processing features with cross-attention.
        regression (Regression): Regression block for predicting rotation and translation parameters.
    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the CalibViTV1 model.
        
        Args:
            config (ModelConfig): Configuration object containing model parameters.
        """
        super(CalibViTV1, self).__init__()
        
        dim = config.dim  # Dimension of embeddings
        mlp_dim = dim * 4  # MLP dimension within transformer blocks

        # Feature extractor for RGB and depth images
        self.feature_extractor = FeatureExtraction(planes=config.num_layers // 8)

        # Linear projection layers for transforming ResNet outputs to ViT inputs
        self.resnet_to_vit = LinearProjection(
            resnet_output_shape=(math.ceil(config.image_size[0] / 8), math.ceil(config.image_size[1] / 8)),
            num_patches=512, embed_dim=dim
        )
        self.depth_resnet_to_vit = LinearProjection(
            resnet_output_shape=(math.ceil(config.image_size[0] / 8), math.ceil(config.image_size[1] / 8)),
            num_patches=256, embed_dim=dim
        )
        
        # Create sinusoidal positional embeddings for the transformer
        self.positional_embeddings = self.create_positional_embeddings(768, dim)
    
        # Cross-attention block (Vision Transformer)
        self.vit = VisionTransformer(
            dim=dim,
            heads=config.ca_heads,
            mlp_dim=mlp_dim,
            dropout=config.ca_layer_dropout,
            drop_path_rate=config.ca_layer_drprate,
            n_crosslayers=config.ca_layers
        )
        
        # Regression block for predicting rotation and translation
        self.regression = Regression(
            in_channels=dim, 
            fc_hidden_layer_size=config.fc_hidden_layer_size,
            dim=dim, 
            fc_layer_dropout=config.fc_layer_dropout
        )

    def forward(self, rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor) -> tuple:
        """
        Forward pass through the C2FNet_V4 model.
        
        Args:
            rgb_tensor (torch.Tensor): Input tensor of RGB images, shape [B, 3, H, W].
            depth_tensor (torch.Tensor): Input tensor of depth images, shape [B, 1, H, W].
        
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Predicted rotation parameters of shape [B, 3].
                - torch.Tensor: Predicted translation parameters of shape [B, 3].
        """
        # Extract features from RGB and depth images using feature extractor
        rgb_activation, depth_activation = self.feature_extractor(rgb_tensor, depth_tensor)  # [B, 512, H/8, W/8], [B, 256, H/8, W/8]

        # Project the extracted features from ResNet to ViT-compatible embeddings
        rgb_final_embeddings = self.resnet_to_vit(rgb_activation)  # [B, 512, 256]
        depth_final_embeddings = self.depth_resnet_to_vit(depth_activation)  # [B, 512, 256]

        # Concatenate RGB and depth embeddings
        final_embeddings = torch.cat((rgb_final_embeddings, depth_final_embeddings), dim=1)
         
        # Add positional embeddings to the final embeddings
        final_embeddings = final_embeddings + self.positional_embeddings.to(final_embeddings.device)

        # Pass the embeddings through the Vision Transformer
        cross_output = self.vit(final_embeddings, final_embeddings)  # [B, 512, 256]

        # Predict rotation and translation using the regression block
        x_rot, x_tr = self.regression(cross_output)  # [B, 3], [B, 3]

        return x_rot, x_tr
    
    def create_positional_embeddings(self, num_patches: int, embed_dim: int) -> torch.Tensor:
        """
        Create sinusoidal positional embeddings for the transformer model.
        
        Args:
            num_patches (int): Number of patches (sequence length).
            embed_dim (int): Dimension of the embeddings (embedding size).

        Returns:
            torch.Tensor: Sinusoidal positional embeddings of shape [1, num_patches, embed_dim].
        """
        positions = torch.arange(num_patches).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)  # Add batch dimension





class CalibViTV2(nn.Module):
    """
    CalibViTV2 is a deep learning model for calibration tasks that processes RGB and depth image data 
    through Swin Transformer backbones and a Vision Transformer (ViT) for feature extraction and regression 
    to predict calibration parameters (rotation and translation).

    Attributes:
        rgb_swin (nn.Module): Swin Transformer model for processing RGB images.
        depth_swin (nn.Module): Swin Transformer model for processing depth images.
        position_embeddings_rgb (nn.Parameter): Learned positional embeddings for the RGB features.
        vit_rgb (VisionTransformer): Vision Transformer model for cross-attention processing of RGB and depth features.
        regression (Regression): Regression block for predicting the rotation and translation parameters.
    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the CalibViTV2 model.
        
        Args:
            config (ModelConfig): Configuration object containing model parameters such as image size, 
                                   number of attention heads, layers, etc.
        """
        super(CalibViTV2, self).__init__()
        
        self.config = config
        self.num_layers = config.in_channels
        self.dim = config.dim  # Dimension of feature map after patch embedding
        self.mlp_dim = self.dim * 4  # MLP dimension for Vision Transformer layers
        
        # Swin Transformer backbone for RGB and Depth images
        self.rgb_swin = timm.create_model(
            'swin_tiny_patch4_window7_224',  # Pretrained Swin Transformer model
            pretrained=True,
            img_size=(320,512),
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
        self.depth_swin = timm.create_model(
            'swin_tiny_patch4_window7_224',  # Pretrained Swin Transformer model
            pretrained=True,
            img_size=(320,512),
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
        
        # Remove the classification head from the Swin models
        self.rgb_swin.head = nn.Identity()
        self.depth_swin.head = nn.Identity()
        
        # Learned positional embeddings for the RGB feature map
        self.position_embeddings_rgb = nn.Parameter(torch.randn(1, 320, 768))
        
        # Vision Transformer (ViT) for cross-attention processing of RGB and depth features
        self.vit_rgb = Transformer(
            dim=self.dim,
            heads=self.config.ca_heads,  # Attention heads per layer
            mlp_dim=self.config.dim,  # MLP hidden dimension for the ViT layers
            dropout=self.config.ca_layer_dropout,
            drop_path_rate=self.config.ca_layer_drprate,
            n_crosslayers=self.config.ca_layers,
            reduction_dims=None  # Optional dimensionality reduction
        )
        
        # Regression network for predicting rotation and translation
        self.regression = Regression(
            in_channels=320,
            fc_hidden_layer_size=self.config.fc_hidden_layer_size,
            dim=self.config.dim,
            fc_layer_dropout=self.config.fc_layer_dropout
        )

        self.apply()  # Apply parameter initialization

    def forward(self, rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CalibViTV2 model.
        
        Args:
            rgb_tensor (torch.Tensor): Input tensor for RGB images of shape [B, 3, H, W].
            depth_tensor (torch.Tensor): Input tensor for depth images of shape [B, 1, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Rotation vector tensor of shape [B, 3].
                - Translation vector tensor of shape [B, 3].
        """
        # Extract features from RGB and depth images using the Swin Transformer backbones
        rgb_activation = self.rgb_swin(rgb_tensor)  # [B, C, H/32, W/32]
        depth_activation = self.depth_swin(depth_tensor)  # [B, C, H/32, W/32]
      
        B, H, W, C = rgb_activation.shape

        # Flatten the feature maps to prepare for attention
        rgb_activation = rgb_activation.view(B, H * W, C)
        depth_activation = depth_activation.view(B, H * W, C)

        # Concatenate RGB and depth feature maps
        activation = torch.cat((rgb_activation, depth_activation), dim=1)
        activation = activation + self.position_embeddings_rgb  # Add positional embeddings
        
        # Pass the concatenated activation through the Vision Transformer
        cross_output_rgb = self.vit_rgb(activation, activation)  # Output shape: [B, 512, 256]
     
        # Pass the transformer output through the regression network
        x_rot, x_tr = self.regression(cross_output_rgb)  # Output shape: [B, 3], [B, 3]
        
        return x_rot, x_tr

    def apply(self):
        """
        Initializes the parameters of the model.
        Ensures that the weights of the Swin Transformer and Vision Transformer backbones are trainable.
        """
        for param in self.rgb_swin.parameters():
            param.requires_grad = True
        
        for param in self.depth_swin.parameters():
            param.requires_grad = True

        for param in self.vit_rgb.parameters():
            param.requires_grad = True



class CalibrationDecoder(nn.Module):

    def __init__(
        self, 
        embedding_dim=768, 
        feature_length=320, 
        num_calibration_tokens=4, 
        num_heads=8, 
        num_decoder_layers=6
    ):
        """
        Initializes the CalibrationDecoder.

        Args:
            embedding_dim (int): Dimension of the embedding space for input features and tokens.
            feature_length (int): Length of the concatenated feature map (number of patches).
            num_calibration_tokens (int): Number of learnable calibration tokens used in the Transformer decoder.
            num_heads (int): Number of attention heads in the Transformer decoder.
            num_decoder_layers (int): Number of layers in the Transformer decoder.
        """
        super(CalibrationDecoder, self).__init__()
        
        # Learnable calibration tokens
        self.calibration_tokens = nn.Parameter(torch.randn(1, num_calibration_tokens, embedding_dim))
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Linear layers for projecting the decoder output to rotation and translation
        self.rotation_projection = nn.Linear(embedding_dim, 3)
        self.translation_projection = nn.Linear(embedding_dim, 3)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CalibrationDecoder.

        Args:
            features (torch.Tensor): Concatenated feature map from RGB and depth data 
                                     with shape (B, feature_length, embedding_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Rotation parameters of shape (B, 3).
                - Translation parameters of shape (B, 3).
        """
        # Get batch size from the input features
        batch_size = features.size(0)

        # Expand calibration tokens to match the batch size
        calibration_tokens = self.calibration_tokens.expand(batch_size, -1, -1)

        # Apply the Transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=calibration_tokens.transpose(0, 1),  # Calibration tokens as queries (T, B, E)
            memory=features.transpose(0, 1)          # Input features as keys and values (S, B, E)
        ).transpose(0, 1)  # Bring output back to (B, T, E)

        # Average the output across all calibration tokens
        decoder_output = decoder_output.mean(dim=1)  # Shape: (B, E)

        # Project the output to rotation and translation parameters
        rotation_output = self.rotation_projection(decoder_output)  # Shape: (B, 3)
        translation_output = self.translation_projection(decoder_output)  # Shape: (B, 3)

        return rotation_output, translation_output
