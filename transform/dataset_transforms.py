import torch 
from .rodrigues import SE3, SO3
from math import pi as PI
from collections.abc import Iterable
from torch.distributions import Normal
from scipy.stats import invgauss

class UniformTransformSE3:
    """
    A class for generating and applying SE3 transformations, with support for different distributions
    for generating random twists (rotations and translations).

    Attributes:
        max_deg (float): Maximum degree for rotation.
        max_tran (float): Maximum translation.
        distribution (str): The distribution to use for generating random transformations ('uniform', 'gaussian', 'inverse_gaussian').
        randomly (bool): Whether to generate random magnitudes for rotation and translation or use maximum values.
        concat (bool): Whether to concatenate the transformed features with original features.
        gt (torch.Tensor): Ground truth transformation matrix.
        igt (torch.Tensor): Inverse ground truth transformation matrix.
        so3 (SO3): SO3 object for handling rotations.
        se3 (SE3): SE3 object for handling transformations.
        count (int): Counter for generating transformations.
    """

    def __init__(
        self, 
        max_deg: float, 
        max_tran: float, 
        distribution: str = 'uniform', 
        mag_randomly: bool = False, 
        concat: bool = False
    ) -> None:
        """
        Initializes the UniformTransformSE3 object.

        Args:
            max_deg (float): Maximum rotation angle in degrees.
            max_tran (float): Maximum translation in meters.
            distribution (str): The distribution for generating random transformations ('uniform', 'gaussian', 'inverse_gaussian').
            mag_randomly (bool): Whether to randomly generate magnitudes for rotation and translation.
            concat (bool): Whether to concatenate the transformed features with the original features.

        Initializes SE3 and SO3 objects for transformation handling.
        """
        self.max_deg = max_deg
        self.max_tran = max_tran
        self.randomly = mag_randomly
        self.concat = concat
        self.distribution = distribution
        self.gt = torch.zeros([1, 4, 4])  # Ground truth transformation matrix
        self.igt = torch.zeros([1, 4, 4])  # Inverse ground truth transformation matrix
        self.so3 = SO3()  # SO3 for rotation transformations
        self.se3 = SE3()  # SE3 for full transformations (rotation + translation)
        self.count = 0

    def generate_transform(self) -> torch.Tensor:
        """
        Generates a random SE3 transformation (rotation and translation) based on the specified distribution.

        Returns:
            torch.Tensor: A twist vector (size [1, 6]) representing the transformation.
        
        Raises:
            NameError: If the distribution is not one of the valid options ('uniform', 'gaussian', 'inverse_gaussian').
        """
        self.count += 1

        # Generate random magnitudes for rotation and translation
        if self.randomly:
            deg = torch.rand(1).item() * self.max_deg
            tran = torch.rand(1).item() * self.max_tran
        else:
            deg = self.max_deg
            tran = self.max_tran

        # Convert degrees to radians for rotation
        amp = deg * torch.pi / 180.0
        
        # Handle different distributions
        if self.distribution == 'uniform':
            # Uniform distribution for rotation and translation
            w = torch.randn(1, 3)
            w = w / w.norm(p=2, dim=1, keepdim=True) * amp * 1.2

            t = torch.randn(1, 3) * tran
            t = t / t.norm(p=2, dim=1, keepdim=True) * tran * 1.2

            # Clipping values to ensure they stay within specified ranges
            w = torch.clamp(w, min=-amp, max=amp)
            t = torch.clamp(t, min=-tran, max=tran)
        
        elif self.distribution == 'gaussian':
            # Gaussian distribution for rotation and translation
            w = (2 * torch.rand(1, 3) - 1) * amp
            t = (2 * torch.rand(1, 3) - 1) * tran
            w = torch.clamp(w, min=-amp, max=amp)
            t = torch.clamp(t, min=-tran, max=tran)
        
        elif self.distribution == 'inverse_gaussian':
            # Inverse Gaussian distribution for rotation and translation
            mu_w = amp  # Mean for rotation
            mu_t = tran  # Mean for translation
            lambda_w = 0.01  # Shape parameter for rotation
            lambda_t = 0.001  # Shape parameter for translation

            invgauss_samples_w = invgauss.rvs(mu=mu_w, scale=lambda_w, size=3)
            invgauss_samples_t = invgauss.rvs(mu=mu_t, scale=lambda_t, size=3)

            # Convert to torch tensors
            w = torch.tensor(invgauss_samples_w, dtype=torch.float32).unsqueeze(0)
            t = torch.tensor(invgauss_samples_t, dtype=torch.float32).unsqueeze(0)

            # Normalize and scale samples
            w = w / w.norm(p=2, dim=1, keepdim=True) * amp
            t = t / t.norm(p=2, dim=1, keepdim=True) * tran

            # Scale rotation and translation to fit desired range
            w = (w - w.min()) / (w.max() - w.min()) * amp
            t = (t - t.min()) / (t.max() - t.min()) * tran

            # Apply random sign flipping to the transformation every other time
            if self.count % 2 == 0:
                t *= -1.0
                w *= -1.0
        
        else:
            raise NameError(f'Invalid distribution {self.distribution}')
        
        # Generate the SE3 transformation matrix from the twist vector
        R = self.so3.exp(w)  # Convert the rotation vector to rotation matrix
        G = torch.zeros(1, 4, 4)
        G[:, 3, 3] = 1
        G[:, 0:3, 0:3] = R
        G[:, 0:3, 3] = t

        # Convert the SE3 matrix to a twist vector
        x = self.se3.log(G)
        return x  # Return the twist vector (size [1, 6])

    def apply_transform(self, p0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the generated transformation to a given tensor.

        Args:
            p0 (torch.Tensor): The input tensor, representing a set of points (size [3, N] or [6, N]).
            x (torch.Tensor): The twist vector representing the transformation (size [1, 6]).

        Returns:
            torch.Tensor: The transformed tensor (size [3, N] or [6, N] depending on concatenation).
        """
        # Apply SE3 transformation
        g = self.se3.exp(x).to(p0)  # Apply the transformation to the input tensor
        gt = self.se3.exp(-x).to(p0)  # Inverse transformation

        self.gt = gt.squeeze(0)  # Ground truth transformation (p1 -> p0)
        self.igt = g.squeeze(0)  # Inverse ground truth transformation (p0 -> p1)

        # Optionally concatenate transformed features
        if self.concat:
            return torch.cat([self.se3.transform(g, p0[:3, :]), self.so3.transform(g[:, :3, :3], p0[3:, :])], dim=1)
        else:
            return self.se3.transform(g, p0)  # Apply the SE3 transformation

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Generates a transformation and applies it to the given tensor.

        Args:
            tensor (torch.Tensor): The input tensor to be transformed.

        Returns:
            torch.Tensor: The transformed tensor after applying the generated transformation.
        """
        x = self.generate_transform()  # Generate the transformation
        return self.apply_transform(tensor, x)  # Apply the transformation to the tensor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Calls the transform method to apply a transformation to the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor to be transformed.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        return self.transform(tensor)

class DepthImgGenerator:
    def __init__(self, img_shape:Iterable, InTran:torch.Tensor, pcd_range:torch.Tensor, intensity:torch.Tensor, density:torch.Tensor, pooling_size=5):
        """
        Initializes the DepthImgGenerator class which will transform point cloud data (pcd) and project it to an image space.
        
        Args:
            img_shape (Iterable): Shape of the image (H, W).
            InTran (torch.Tensor): Intrinsic matrix (3x3 or 4x4). This is the camera's intrinsic parameters.
            pcd_range (torch.Tensor): The range of the point cloud. This is used to represent the depth of points.
            intensity (torch.Tensor): The intensity values of the points in the point cloud.
            density (torch.Tensor): The density values of the points in the point cloud.
            pooling_size (int, optional): Pooling size for the max pooling operation to reduce the image size. Defaults to 5.

        Raises:
            AssertionError: If pooling_size is not odd.
        """
        # Ensure that the pooling size is odd for the size to remain consistent after pooling
        assert (pooling_size - 1) % 2 == 0, 'pooling size must be odd to keep image size constant'
        
        # Initialize max pooling layer to reduce the image size
        self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=(pooling_size - 1) // 2)
        
        # Initialize intrinsic matrix (InTran) as a 3x3 identity matrix by default
        self.img_shape = img_shape
        self.InTran = torch.eye(3)[None, ...]  # Create identity matrix of shape [1, 3, 3]
        self.InTran[0, :InTran.size(0), :InTran.size(1)] = InTran  # Update with given intrinsic matrix
        
        # Store the point cloud range, intensity, and density for each point in the point cloud
        self.pcd_range = pcd_range  # (B, N)
        self.intensity = intensity
        self.density = density
        
        # Initialize SE3 transformation object for 3D transformations
        self.se3 = SE3()

    def transform(self, ExTran:torch.Tensor, pcd:torch.Tensor) -> tuple:
        """
        Transforms the point cloud (pcd) and projects it onto an image.
        
        Args:
            ExTran (torch.Tensor): A batch of transformation matrices (B, 4, 4).
            pcd (torch.Tensor): A batch of point clouds (B, 3, N), where N is the number of points in each cloud.
        
        Returns:
            tuple: 
                - depth_img (B, H, W): The depth image with the projected depth, intensity, and density.
                - transformed_pcd (B, 3, N): The transformed point clouds after applying ExTran.
        """
        # Unpack image height and width from img_shape
        H, W = self.img_shape
        B = ExTran.size(0)  # Batch size
        
        # Ensure the intrinsic matrix is on the same device as the point cloud
        self.InTran = self.InTran.to(pcd.device)
        
        # Apply the given transformation to the point cloud
        pcd = self.se3.transform(ExTran, pcd)  # [B, 4, 4] x [B, 3, N] -> [B, 3, N]
        
        # Project the point cloud to image space using the intrinsic matrix
        proj_pcd = torch.bmm(self.InTran.repeat(B, 1, 1), pcd)  # [B, 3, 3] x [B, 3, N] -> [B, 3, N]
        
        # Convert to pixel coordinates (x, y) in image space
        proj_x = (proj_pcd[:, 0, :] / proj_pcd[:, 2, :]).type(torch.long)
        proj_y = (proj_pcd[:, 1, :] / proj_pcd[:, 2, :]).type(torch.long)
        
        # Apply visibility checks to keep only valid points that are within the image bounds
        rev = ((proj_x >= 0) * (proj_x < W) * (proj_y >= 0) * (proj_y < H) * (proj_pcd[:, 2, :] > 0)).type(torch.bool)
        
        # Initialize an empty depth image to store the depth, intensity, and density for each point
        batch_depth_img = torch.zeros(B, 3, H, W, dtype=torch.float32).to(pcd.device)
        
        # Loop over each batch to fill in the depth image
        for bi in range(B):
            rev_i = rev[bi, :]  # Get visibility mask for the current batch
            
            # Get the valid point indices
            proj_xrev = proj_x[bi, rev_i]
            proj_yrev = proj_y[bi, rev_i]
            
            # Fill in depth, intensity, and density values for valid pixels
            batch_depth_img[bi, 0, proj_yrev, proj_xrev] = self.pcd_range[bi, rev_i]  # Depth
            batch_depth_img[bi, 1, proj_yrev, proj_xrev] = self.intensity[bi, rev_i]  # Intensity
            batch_depth_img[bi, 2, proj_yrev, proj_xrev] = self.density[bi, rev_i]  # Density
        
        # Apply max pooling to reduce the image size
        batch_depth_img = self.pooling(batch_depth_img)
        
        return batch_depth_img, pcd  # Return the generated depth image and transformed point cloud
    
    def __call__(self, ExTran:torch.Tensor, pcd:torch.Tensor):
        """
        A callable function to transform the point cloud and project it onto the image.
        
        Args:
            ExTran (torch.Tensor): A batch of transformation matrices (B, 4, 4).
            pcd (torch.Tensor): A batch of point clouds (B, 3, N).
        
        Returns:
            tuple: 
                - depth_img (B, H, W): The depth image with the projected data.
                - transformed_pcd (B, 3, N): The transformed point cloud.
        """
        # Ensure the correct shape for ExTran and pcd
        assert len(ExTran.size()) == 3, 'ExTran size must be (B, 4, 4)'
        assert len(pcd.size()) == 3, 'pcd size must be (B, 3, N)'
        
        # Call the transform method to generate the depth image
        return self.transform(ExTran, pcd)
