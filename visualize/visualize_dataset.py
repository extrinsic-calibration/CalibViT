from dataloader.dataset_utils import PointCloudProjection
import numpy as np 
from typing import Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os 

class VisualizeDataset:
    """Class to visualize projected points from point clouds onto RGB images and depth images."""

    @staticmethod
    def plot_projected_points_on_image(
        data: Dict, 
        save_path: Optional[str] = None,  
        all_points: bool = True, 
        marker_size: int = 5
    ) -> None:
        """
        Plot projected points from a point cloud onto an RGB image.

        Parameters:
            data (dict): A dictionary containing the following keys:
                - 'InTran': Intrinsic matrix of the camera (shape: (1, 3, 3)).
                - 'uncalibed_pcd': Uncalibrated point cloud data (shape: (1, N, 3)).
                - 'pcd': Calibrated point cloud data (shape: (1, N, 3)).
                - 'intensity': Intensity values associated with the point cloud (shape: (1, N)).
                - 'img': RGB image (shape: (1, C, H, W)).
            all_points (bool): If True, project all points. If False, only project points within the image bounds.
            marker_size (int): Size of the markers in the scatter plot.
            save_path (str, optional): Path to save the image. If None, the image is not saved.

        Returns:
            None: This function does not return any value. It generates visualizations.
        """
        # Get intrinsic matrix from data
        intrinsic_matrix = data['InTran'].squeeze(0).numpy()

        # Get uncalibrated point cloud and calibrated point cloud, and intensity values
        point_cloud_uncalib = data['uncalibed_pcd'].squeeze(0).numpy().T
        point_cloud_calib = data['pcd'].squeeze(0).numpy().T
        intensity = data['intensity'].squeeze(0).numpy()

        # Get RGB image from data
        image = data['img'].squeeze(0).numpy().transpose(1, 2, 0)

        # Initialize the point cloud projector
        projector = PointCloudProjection()

        if not all_points:
            # Get only the points that fall within the image bounds for uncalibrated point cloud
            _, _, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_uncalib.T)
            point_cloud_uncalib = point_cloud_uncalib[r, :]

            # Get only the points that fall within the image bounds for calibrated point cloud
            _, _, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_calib.T)
            point_cloud_calib = point_cloud_calib[r, :]
            intensity_calib = intensity[r]

        # Project uncalibrated points using the intrinsic matrix
        projected_points_uncalib = np.dot(intrinsic_matrix, point_cloud_uncalib.T).T
        projected_points_uncalib /= projected_points_uncalib[:, 2].reshape(-1, 1)  # Normalize by the z-coordinate

        # Project calibrated points using the intrinsic matrix
        projected_points_calib = np.dot(intrinsic_matrix, point_cloud_calib.T).T
        projected_points_calib /= projected_points_calib[:, 2].reshape(-1, 1)  # Normalize by the z-coordinate
        
        # Color the points based on their depth using a colormap
        coloring_uncalib = point_cloud_uncalib[:, 2]
        coloring_calib = point_cloud_calib[:, 2]
        
        # Normalize depth values for colormap
        norm_uncalib = (coloring_uncalib - np.min(coloring_uncalib)) / (np.max(coloring_uncalib) - np.min(coloring_uncalib))
        norm_calib = (coloring_calib - np.min(coloring_calib)) / (np.max(coloring_calib) - np.min(coloring_calib))
        
        # Create colormaps
        cmap = cm.viridis

        # Plot the projected points on the image for uncalibrated data
        fig = plt.figure(figsize=(16, 9))
        plt.imshow(image)  # Display the RGB image
        scatter_uncalib = plt.scatter(
            projected_points_uncalib[:, 0], 
            projected_points_uncalib[:, 1], 
            c=cmap(norm_uncalib), 
            s=marker_size, 
            alpha=0.6, 
            edgecolor='none'
        )
        plt.title('Projected Points on Image (Decalibrated)', fontsize=16)
        plt.axis('off')  # Hide the axis
        plt.show()  # Show the plot
        if save_path:
            plt.savefig(os.path.join(save_path, "projected_points_decalib.png"), dpi=300)

        # Plot the projected points on the image for calibrated data
        plt.figure(figsize=(16, 9))
        plt.imshow(image)  # Display the RGB image
        scatter_calib = plt.scatter(
            projected_points_calib[:, 0], 
            projected_points_calib[:, 1], 
            c=cmap(norm_calib), 
            s=marker_size, 
            alpha=0.6, 
            edgecolor='none'
        )
        plt.title('Projected Points on Image (Ground Truth)', fontsize=16)
        plt.axis('off')  # Hide the axis
        plt.show()  # Show the plot

    @staticmethod
    def plot_projected_points_decalib(
        data: Dict, 
        all_points: bool = True
    ) -> None:
        """
        Plot the projected points from an uncalibrated point cloud onto an RGB image.

        Parameters:
            data (dict): A dictionary containing the following keys:
                - 'InTran': Intrinsic matrix of the camera (shape: (1, 3, 3)).
                - 'uncalibed_pcd': Uncalibrated point cloud data (shape: (1, N, 3)).
                - 'pcd': Calibrated point cloud data (shape: (1, N, 3)).
                - 'img': RGB image (shape: (1, C, H, W)).
            all_points (bool): If True, project all points. If False, only project points within the image bounds.

        Returns:
            None: This function does not return any value. It generates visualizations.
        """
        # Get intrinsic matrix
        intrinsic_matrix = data['InTran'].squeeze(0).numpy()

        # Get uncalibrated point cloud and calibrated point cloud
        point_cloud_uncalib = data['uncalibed_pcd'].squeeze(0).numpy().T
        point_cloud_calib = data['pcd'].squeeze(0).numpy().T

        # RGB image
        image = data['img'].squeeze(0).numpy().transpose(1, 2, 0)

        # Initialize point cloud projector
        projector = PointCloudProjection()

        if not all_points:
            # Get only the points that fall within the image bounds for uncalibrated point cloud
            _, _, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_uncalib.T)
            point_cloud_uncalib = point_cloud_uncalib[r, :]

            # Get only the points that fall within the image bounds for calibrated point cloud
            _, _, r = projector.binary_projection(img_shape=image.shape[:2], intrinsic=intrinsic_matrix, pcd=point_cloud_calib.T)
            point_cloud_calib = point_cloud_calib[r, :]

        # Project uncalibrated points using the intrinsic matrix
        projected_points_uncalib = np.dot(intrinsic_matrix, point_cloud_uncalib.T).T
        projected_points_uncalib /= projected_points_uncalib[:, 2].reshape(-1, 1)  # Normalize by the z-coordinate

        # Project calibrated points using the intrinsic matrix
        projected_points_calib = np.dot(intrinsic_matrix, point_cloud_calib.T).T
        projected_points_calib /= projected_points_calib[:, 2].reshape(-1, 1)  # Normalize by the z-coordinate

        # Depth-wise coloring
        coloring_uncalib = point_cloud_uncalib[:, 2]
        coloring_calib = point_cloud_calib[:, 2]

        # Plot the projected points on the image
        plt.figure(figsize=(16, 16), dpi=300)
        plt.imshow(image)  # Display the RGB image
        plt.scatter(projected_points_uncalib[:, 0], projected_points_uncalib[:, 1], c=coloring_uncalib, s=10)
        plt.title('Initial Decalibration', fontsize=16)
        plt.axis('off')  # Hide the axis
        plt.show()

    @staticmethod
    def plot_depth_image(
        data: Dict, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot calibrated and uncalibrated depth images and optionally save the figure.

        Parameters:
            data (dict): A dictionary containing the following keys:
                - 'depth_img': Calibrated depth image (shape: (1, H, W)).
                - 'uncalibed_depth_img': Uncalibrated depth image (shape: (1, H, W)).
            save_path (str, optional): Path to save the image. If None, the image is not saved.

        Returns:
            None: This function does not return any value. It generates visualizations.
        """
        # Get the depth images
        depth_img = data['depth_img'].squeeze(0).numpy()
        uncalibed_depth_img = data['uncalibed_depth_img'].squeeze(0).numpy()

        # Plot the depth images
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(uncalibed_depth_img, cmap='plasma')
        plt.title('Uncalibrated Depth Image', fontsize=16)
        plt.axis('off')  # Hide the axis

        plt.subplot(1, 2, 2)
        plt.imshow(depth_img, cmap='plasma')
        plt.title('Calibrated Depth Image', fontsize=16)
        plt.axis('off')  # Hide the axis

        # Optionally save the figure
        if save_path:
            plt.savefig(os.path.join(save_path, "depth_images.png"), dpi=300)
        plt.show()