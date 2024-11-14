import os
import numpy as np
from typing import Any, Tuple, Dict, Optional
import json 
from config import Config
# torch 
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms as Tf

# Image
from PIL import Image

# Nuscenes 
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

# Tools
from .dataset_utils import ToTensor, PointCloudFilter, PointCloudProjection, PointCloudResampler, MinMaxScaler, PointCloudDensity
from transform import UniformTransformSE3
from transform import SE3, SO3


class NuScenesLoader:
    def __call__(self, data_root: str, version:str ='v1.0-trainval',  verbose: bool = False) -> NuScenes:
        """Load the NuScenes dataset.

        Args:
            data_root (str): Path to the NuScenes data directory.
            version (str, optional): NuScenes dataset version (default is 'v1.0-trainval').
            verbose (bool, optional): Whether to display verbose information. Defaults to False.

        Returns:
            NuScenes: NuScenes dataset instance.
        """
        return NuScenes(version=version, dataroot=data_root, verbose=verbose)

class NuScenesDataset(Dataset):
    def __init__(self, nusc: NuScenes, config: Config, split: str = 'train', limscenes: Optional[int] = None) -> None:
        """
        Initialize the NuScenes dataset.

        Args:
            nusc (NuScenes): NuScenes API instance for data retrieval.
            config (Config): Configuration object containing various dataset parameters.
            split (str, optional): Dataset split ('train', 'val', or 'test'). Defaults to 'train'.
            limscenes (Optional[int], optional): Limit the number of scenes to use. If None, all scenes are used.

        """
        # Store config and NuScenes API instance
        self.config = config
        self.nusc = nusc

        # Get the scene tokens based on the split
        self.scene_tokens = self.get_scene_tokens(split)

        # Optionally limit the number of scenes to process
        # if limscenes is not None:
        #     self.scene_tokens = self.scene_tokens[:limscenes]

        # Get all sample tokens for the scenes
        self.sample_tokens = self.get_sample_tokens()

        # Initialize transformation and data processing tools
        self.np_to_tensor = ToTensor(tensor_type=torch.float)
        self.img_to_tensor = Tf.ToTensor()
        self.point_cloud_sampler = PointCloudResampler(num_points=self.config.dataset_config.pcd_min_samples)
        self.point_cloud_filter = PointCloudFilter(range_threshold=self.config.dataset_config.max_depth, min_neighbors=2, concat='none')
        self.pt_projection = PointCloudProjection()
        self.get_desnsity = PointCloudDensity(radius=0.3)
        self.scaler_desnity = MinMaxScaler(min_val=0, max_val=30)
        self.scaler_range = MinMaxScaler(min_val=self.config.dataset_config.min_depth, max_val=self.config.dataset_config.max_depth)
        self.scaler_intensity = MinMaxScaler(min_val=self.config.dataset_config.min_intensity, max_val=self.config.dataset_config.max_intensity)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a specific sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing data for the selected sample, including the image, point cloud, intrinsic/extrinsic matrices, and depth image.
        """
        # Get the sample token at the given index
        sample_token = self.sample_tokens[idx]

        # Retrieve sample data from NuScenes API
        sample = self.nusc.get('sample', sample_token)

        # Get camera and LiDAR tokens
        cam_front_token = sample['data']['CAM_FRONT']
        lidar_token = sample['data']['LIDAR_TOP']

        # Load image, calibration matrix and transformation matrix
        image, image_size, adjustment_ratio = self.load_image(token=cam_front_token)
        intrinsic_resized, intrinsic_extended = self.get_intrinsic_matrix(cam_token=cam_front_token, adjustment_ratio=adjustment_ratio)
        extrinsic = self.get_extrinsic_matrix(lidar_token=lidar_token, cam_token=cam_front_token)

        # Load lidar point cloud data and project to image space
        point_cloud, intensity = self.load_lidar_point_cloud(token=lidar_token,
                                                 cam_token=cam_front_token,
                                                 intrinsic_matrix=intrinsic_extended,
                                                 extrinsic_matrix=extrinsic,
                                                 image_size=image_size)

        # Generate depth image, range, intensity, and density maps
        depth_image, pcd_range, intensity, density = self.get_depth_image(point_cloud=point_cloud,
                                                     image_size=image_size,
                                                     intrinsic_matrix=intrinsic_resized,
                                                     intensity=intensity)

        # Return sample data as a dictionary
        return dict(
            img=image,
            pcd=self.np_to_tensor(point_cloud),
            pcd_range=pcd_range,
            intensity=intensity,
            density=density,
            depth_img=depth_image,
            InTran=self.np_to_tensor(intrinsic_resized),
            ExTran=self.np_to_tensor(extrinsic)
        )

    def get_scene_tokens(self, split: str) -> list:
        """
        Get scene tokens based on the dataset split.

        Args:
            split (str): Dataset split ('train', 'val', or 'test').

        Returns:
            list: List of scene tokens.
        """
        # Get predefined scene splits (assumes 'create_splits_scenes' is defined elsewhere)
        scene_splits = create_splits_scenes()

        # Select the appropriate scene tokens based on the split
        if split in ['train', 'val']:
            scene_splits[split] = [self.nusc.field2token('scene', 'name', scene_name)[0] for scene_name in scene_splits[split]]
        elif split == 'test':
            scene_splits['test'] = [self.nusc.field2token('scene', 'name', scene_name)[0] for scene_name in scene_splits['test']]

        return scene_splits[split]

    def get_sample_tokens(self) -> list:
        """
        Get a list of sample tokens for all scenes.

        Returns:
            list: List of sample tokens across all scenes.
        """
        sample_tokens = []
        for scene_token in self.scene_tokens:
            sample_tokens.extend(self.get_sample_token(scene_token=scene_token))
        return sample_tokens

    def get_sample_token(self, scene_token: str) -> list:
        """
        Get sample tokens for a specific scene.

        Args:
            scene_token (str): Token of the scene.

        Returns:
            list: List of sample tokens within the scene.
        """
        sample_tokens = []
        scene = self.nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        while sample_token:
            sample_tokens.append(sample_token)
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']

        return sample_tokens

    def load_image(self, token: str) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
        """
        Load and preprocess an image from the dataset.

        Args:
            token (str): Token of the sample data.

        Returns:
            Tuple[torch.Tensor, Tuple[int, int], np.ndarray]: A tuple containing the preprocessed image as a PyTorch tensor, the resized image dimensions, and an adjustment matrix.
        """
        # Retrieve image data for the given token
        cam_data = self.nusc.get('sample_data', token)
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        H, W = image.height, image.width
        adjustment_ratio = np.diag([self.config.dataset_config.image_size[1]/W, self.config.dataset_config.image_size[0]/H, 1])
        image = image.resize((self.config.dataset_config.image_size[1], self.config.dataset_config.image_size[0]), Image.BILINEAR)

        resized_height = round(image.height * self.config.dataset_config.resize_ratio[0])
        resized_width = round(image.width * self.config.dataset_config.resize_ratio[1])
        image = image.resize((resized_width, resized_height), Image.BILINEAR)

        # Convert image to tensor and return
        image_tensor = self.img_to_tensor(image)
        return image_tensor, (resized_height, resized_width), adjustment_ratio


    def get_depth_image(self, lidar_token: str, cam_token: str) -> np.ndarray:
        """
        Retrieve the depth image corresponding to a lidar point cloud in the camera coordinate system.

        Args:
            lidar_token (str): Token of the lidar sample associated with the point cloud.
            cam_token (str): Token of the camera associated with the target image.

        Returns:
            np.ndarray: A 2D depth image (H x W) where each pixel contains the distance from the camera to the point.
        """
        # Get lidar points
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_pointsensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_points = self.nusc.get('lidar', lidar_data['token'])

        # Transformation matrix from lidar frame to camera frame
        transform_lidar_to_camera = self.get_extrinsic_matrix(lidar_token, cam_token)

        # Project the lidar points to the image plane of the camera
        points = np.array(lidar_points['points'])
        points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))  # Add homogeneous coordinate
        points_camera_frame = transform_lidar_to_camera @ points_homogeneous.T  # Apply transformation

        # Convert from camera coordinates to pixel coordinates
        intrinsic_matrix, _ = self.get_intrinsic_matrix(cam_token, np.eye(3))  # Assuming no resizing for simplicity
        pixel_coords = intrinsic_matrix @ points_camera_frame[:3, :]  # Project into image space
        pixel_coords /= pixel_coords[2, :]  # Normalize to get (x, y) in image coordinates

        # Convert to integer pixel indices
        height, width = self.config.dataset_config.height, self.config.dataset_config.width
        pixel_coords = pixel_coords.T  # Transpose to get rows and columns
        valid_points = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)

        # Generate depth image
        depth_image = np.zeros((height, width), dtype=np.float32)
        depth_image[pixel_coords[valid_points, 1].astype(int), pixel_coords[valid_points, 0].astype(int)] = points_camera_frame[2, valid_points]

        return depth_image


    def load_lidar_point_cloud(self, token: str, cam_token: str, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray, image_size: tuple, radius: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a lidar point cloud from the dataset.

        Args:
            token (str): Token of the sample data.
            cam_token (str): Token of the camera data.
            intrinsic_matrix (np.ndarray): Intrinsic matrix for the camera.
            extrinsic_matrix (np.ndarray): Extrinsic matrix for the lidar.
            image_size (tuple): Tuple of (height, width) for the binary image.
            radius (float, optional): Radius for removing close points. Defaults to 0.1.

        Returns:
            np.ndarray: Preprocessed lidar point cloud as a NumPy array.
        """
        # Get path
        lidar_data = self.nusc.get('sample_data', token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])

        # Read pcd.bin file for lidar
        point_cloud = LidarPointCloud.from_file(lidar_path)
        intensity = np.array(point_cloud.points[3, :])

        # Transform point cloud
        point_cloud.transform(extrinsic_matrix)

        # Get points
        points = point_cloud.points[:3, :]

        # Filter points using self.point_cloud_filter method (assuming it's defined elsewhere)
        points, intensity = self.point_cloud_filter(points.T, intensity.T)

        # Project point cloud on binary image and find the valid points (binary_projection)
        height, width = image_size
        height_extended, width_extended = height * self.config.dataset_config.extend_ratio[0], width * self.config.dataset_config.extend_ratio[1]

        *_, rev = self.pt_projection.binary_projection(img_shape=(height_extended, width_extended), intrinsic=intrinsic_matrix, pcd=points.T)
        points = points[rev,:]
        intensity = intensity[rev]
  
        # Subsample the points to keep uniform dimensions 
        points, intensity = self.point_cloud_sampler(points, intensity)

        return points.T, intensity.T 

    def get_intrinsic_matrix(self, cam_token: str, adjustment_ratio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the calibration matrix for a specific camera and its extended version.

        Args:
            cam_token (str): Token of the camera data.
            adjustment_ratio (np.ndarray): Adjustment matrix for resizing or re-scaling.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the original calibration matrix and extended calibration matrix as numpy arrays.
        """
        # Retrieve camera data and corresponding calibrated sensor data
        cam_data = self.nusc.get('sample_data', cam_token)
        sensor_data = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        # Obtain the camera's intrinsic calibration matrix
        calibration_matrix = np.array(sensor_data['camera_intrinsic'])
        calibration_matrix = adjustment_ratio @ calibration_matrix

        # Adjust calibration matrix according to resize ratio
        calibration_matrix = np.diag([self.config.dataset_config.resize_ratio[1], self.config.dataset_config.resize_ratio[0], 1]) @ calibration_matrix

        # Create an extended calibration matrix by applying extend_ratio to translation components
        calibration_matrix_extended = calibration_matrix.copy()
        calibration_matrix_extended[0, -1] *= self.config.dataset_config.extend_ratio[0]
        calibration_matrix_extended[1, -1] *= self.config.dataset_config.extend_ratio[1]

        return calibration_matrix, calibration_matrix_extended


    def get_extrinsic_matrix(self, lidar_token: str, cam_token: str) -> np.ndarray:
        """
        Compute the cumulative transformation matrix to transform a point cloud from the lidar frame to the camera frame.

        Args:
            lidar_token (str): Token of the lidar sample associated with the point cloud.
            cam_token (str): Token of the camera associated with the target image.

        Returns:
            np.ndarray: A 4x4 transformation matrix that captures the cumulative transformation process.
        """
        # Get information about the lidar sample
        pointsensor = self.nusc.get('sample_data', lidar_token)

        # Get information about the camera
        cam = self.nusc.get('sample_data', cam_token)

        # Transformation matrix for the first step: Sensor frame to ego vehicle frame at the sweep timestamp
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        transform = np.eye(4)
        transform[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        transform[:3, 3] = np.array(cs_record['translation'])

        # Transformation matrix for the second step: Ego vehicle frame at the sweep timestamp to global frame
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        ego_to_global = np.eye(4)
        ego_to_global[:3, :3] = Quaternion(poserecord['rotation']).rotation_matrix
        ego_to_global[:3, 3] = np.array(poserecord['translation'])
        transform = np.dot(ego_to_global, transform)

        # Transformation matrix for the third step: Global frame to ego vehicle frame at the image timestamp
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        global_to_ego = np.eye(4)
        global_to_ego[:3, :3] = Quaternion(poserecord['rotation']).rotation_matrix
        global_to_ego[:3, 3] = np.array(poserecord['translation'])
        global_to_ego = np.linalg.inv(global_to_ego)
        transform = np.dot(global_to_ego, transform)

        # Transformation matrix for the fourth step: Ego vehicle frame to camera frame
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        ego_to_cam = np.eye(4)
        ego_to_cam[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        ego_to_cam[:3, 3] = np.array(cs_record['translation'])
        ego_to_cam = np.linalg.inv(ego_to_cam)
        transform = np.dot(ego_to_cam, transform)

        return transform