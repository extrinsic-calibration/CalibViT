# torch 
import torch
from torch.utils.data.dataset import Dataset 
from torchvision.transforms import transforms as Tf

# kitti
import pykitti

# Image
from PIL import Image

# Tools
from .dataset_utils import ToTensor, PointCloudFilter, PointCloudProjection, PointCloudResampler, MinMaxScaler, PointCloudDensity
from transform import UniformTransformSE3
from transform import SE3, SO3

# extras 
import os
import json
from config import Config
import numpy as np 
from typing import Tuple



class KittiDataset(Dataset):
    """
    Custom Dataset class for loading KITTI data, including point cloud, images, intrinsic and extrinsic parameters.
    """

    def __init__(self, config: Config, split: str = 'train'):
        """
        Initializes the dataset by loading sequences, point clouds, images, and other related metadata.
        
        Args:
            config (Config): Configuration object containing dataset paths and parameters.
            split (str, optional): Data split to load ('train', 'val', or 'test'). Default is 'train'.
        """
        self.config = config
        self.meta_json = os.path.join(self.config.dataset_path, "meta_json.json")
        
        # Check for metadata file and create it if it doesn't exist
        if not os.path.exists(self.meta_json):
            self.__check_length()

        # Load metadata
        with open(self.meta_json, 'r') as f:
            dict_len = json.load(f)
        
        # Select the sequences based on the split
        if split == 'train':
            sequences = self.config.dataset_config.train_sequences
        elif split == 'val':
            sequences = self.config.dataset_config.val_sequences
        elif split == 'test':
            sequences = self.config.dataset_config.test_sequences

        # Get frame list, skipping frames according to batch size
        frame_list = []
        for seq in sequences:
            frame = list(range(0, dict_len[seq]))  # Skipping frames functionality can be added here
            frame_list.append(frame)

        # Load KITTI data for the selected sequences
        self.kitti_datalist = [pykitti.odometry(self.config.dataset_path, seq, frames=frame) 
                               for seq, frame in zip(sequences, frame_list)]
        
        # Check if camera frames correspond to LiDAR frames
        for seq, obj in zip(sequences, self.kitti_datalist):
            self.__check(obj, self.config.dataset_config.cam_id, seq)

        # Compute total number of data points
        self.sep = [len(data) for data in self.kitti_datalist]
        self.sumsep = np.cumsum(self.sep)

        # Initialize preprocessing and transformation tools
        self.np_to_tensor = ToTensor(tensor_type=torch.float)
        self.img_to_tensor = Tf.ToTensor()
        self.point_cloud_sampler = PointCloudResampler(num_points=self.config.dataset_config.pcd_min_samples)
        self.point_cloud_filter = PointCloudFilter(range_threshold=self.config.dataset_config.max_depth, 
                                                   min_neighbors=2, concat='none')
        self.pt_projection = PointCloudProjection()
        self.get_desnsity = PointCloudDensity(radius=0.3) 
        self.scaler_desnity = MinMaxScaler(min_val=0, max_val=self.config.dataset_config.num_neighbours)
        self.scaler_range = MinMaxScaler(min_val=self.config.dataset_config.min_depth, 
                                          max_val=self.config.dataset_config.max_depth)
        self.scaler_intensity = MinMaxScaler(min_val=self.config.dataset_config.min_intensity, 
                                             max_val=self.config.dataset_config.max_intensity)

    def __check_length(self):
        """
        Checks the number of LiDAR point clouds per sequence and saves the count in the metadata file.
        """
        seq_dir = os.path.join(self.config.dataset_path, 'sequences')
        seq_list = os.listdir(seq_dir)
        seq_list.sort()
        dict_len = dict()
        
        # Calculate the number of LiDAR points for each sequence
        for seq in seq_list:
            len_velo = len(os.listdir(os.path.join(seq_dir, seq, 'velodyne')))
            dict_len[seq] = len_velo
        
        # Save sequence lengths to the metadata file
        with open(self.meta_json, 'w') as f:
            json.dump(dict_len, f)

    def __len__(self):
        """
        Returns the total number of data points in the dataset.
        
        Returns:
            int: Total number of data points in the dataset.
        """
        return self.sumsep[-1]

    @staticmethod
    def __check(odom_obj: pykitti.odometry, cam_id: int, seq: str) -> bool:
        """
        Verifies the validity of the odometry data by checking camera and LiDAR file consistency.
        
        Args:
            odom_obj (pykitti.odometry): KITTI odometry object.
            cam_id (int): Camera ID for image loading.
            seq (str): Sequence identifier.
        
        Returns:
            bool: Returns True if the data is valid, otherwise raises an assertion error.
        """
        calib = odom_obj.calib
        cam_files_length = len(getattr(odom_obj, 'cam%d_files' % cam_id))
        velo_files_length = len(odom_obj.velo_files)
        head_msg = '[Seq %s]:' % seq
        
        assert cam_files_length > 0, head_msg + ' None of camera %d files' % cam_id
        assert cam_files_length == velo_files_length, head_msg + " Camera %d (%d) and LiDAR files (%d) don't match!" % (cam_id, cam_files_length, velo_files_length)
        assert hasattr(calib, 'T_cam0_velo'), head_msg + " Crucial calibration attribute 'T_cam0_velo' is missing!"
    
    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a data point from the dataset given an index.

        Args:
            index (int): Index of the data point to retrieve.

        Returns:
            dict: Dictionary containing the image, point cloud, range, intensity, density, depth image,
                  intrinsic transformation, and extrinsic transformation.
        """
        # Get group ID and data
        group_id = np.digitize(index, self.sumsep, right=False)
        data = self.kitti_datalist[group_id]

        # Get sub-index within the group
        sub_index = index - self.sumsep[group_id - 1] if group_id > 0 else index

        # Retrieve image, intrinsic/extrinsic parameters, and point cloud
        image, image_size, adjustment_ratio = self.__get_image(data=data, sub_index=sub_index)

        extrinsic = self.__get_extrinsic(data=data)
        intrinsic_resized, intrinsic_extended = self.__get_intrinsic(data=data, adjustment_ratio=adjustment_ratio)

        # Get point cloud and process it
        point_cloud, intensity = self.__get_point_cloud(data=data, sub_index=sub_index, extrinsic=extrinsic, 
                                                        image_size=image_size, intrinsic_matrix=intrinsic_extended)

        # Generate depth and range images
        depth_img, pcd_range_tensor, intensity_tensor, density_tensor = self.__get_depth_image(point_cloud=point_cloud,
                                                                                                image_size=image_size, 
                                                                                                intrinsic_matrix=intrinsic_resized, 
                                                                                                intensity=intensity)

        # Return results as a dictionary
        return {
            'img': image,
            'pcd': self.np_to_tensor(point_cloud),
            'pcd_range': pcd_range_tensor,
            'intensity': intensity_tensor,
            'density': density_tensor,
            'depth_img': depth_img,
            'InTran': self.np_to_tensor(intrinsic_resized),
            'ExTran': self.np_to_tensor(extrinsic)
        }

    def __get_extrinsic(self, data) -> np.ndarray:
        """
        Retrieves the extrinsic transformation matrix (camera to LiDAR) for a given data point.

        Args:
            data: KITTI odometry data object.
        
        Returns:
            np.ndarray: Extrinsic transformation matrix (4x4).
        """
        T_cam2velo = getattr(data.calib, 'T_cam%d_velo' % self.config.dataset_config.cam_id)
        return T_cam2velo

    def __get_intrinsic(self, data, adjustment_ratio) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves and adjusts the intrinsic matrix for a given camera, considering resizing and extension factors.

        Args:
            data: KITTI odometry data object.
            adjustment_ratio: Adjusting scaling factor for the intrinsic matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Adjusted intrinsic matrix and extended intrinsic matrix.
        """
        K_cam = adjustment_ratio @ getattr(data.calib, 'K_cam%d' % self.config.dataset_config.cam_id)
        K_cam = np.diag([self.config.dataset_config.resize_ratio[1], self.config.dataset_config.resize_ratio[0], 1]) @ K_cam
        K_cam_extend = K_cam.copy()
        K_cam_extend[0, -1] *= self.config.dataset_config.extend_ratio[0]
        K_cam_extend[1, -1] *= self.config.dataset_config.extend_ratio[1]
        return K_cam, K_cam_extend

    def __get_image(self, data, sub_index: int) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
        """
        Retrieves and processes the camera image for a given index.

        Args:
            data: KITTI odometry data object.
            sub_index (int): Index of the image within the sequence.

        Returns:
            Tuple[torch.Tensor, Tuple[int, int], np.ndarray]: Processed image tensor, image size (height, width), 
                                                              and adjustment ratio for resizing.
        """
        
        raw_img = getattr(data, 'get_cam%d' % self.config.dataset_config.cam_id)(sub_index)  # PIL Image
        H, W = raw_img.height, raw_img.width
        adjustment_ratio = np.diag([self.config.dataset_config.image_size[1] / W, self.config.dataset_config.image_size[0] / H, 1])
        raw_img = raw_img.resize([self.config.dataset_config.image_size[1], self.config.dataset_config.image_size[0]], Image.BILINEAR)

        H, W = raw_img.height, raw_img.width
        RH = round(H * self.config.dataset_config.resize_ratio[0])
        RW = round(W * self.config.dataset_config.resize_ratio[1])
        
        # Resize and conver to tensor
        raw_img = raw_img.resize([RW, RH], Image.BILINEAR)
        img = self.img_to_tensor(raw_img)  # raw img input (3,H,W)
        return img, (RH, RW), adjustment_ratio

    def __get_point_cloud(self,data, sub_index, extrinsic, image_size, intrinsic_matrix):
        # Get points and intensity 
        pcd = data.get_velo(sub_index)
        intensity = pcd[:,3]
        
        points = pcd[:,:3] # (N,4)

        # Transfrom the point cloud and filter 
        points = points @ extrinsic[:3,:3].T + extrinsic[:3,3]  #points @ points.T  # [4,4] @ [4,N] -> [4,N]
        points, intensity = self.point_cloud_filter(points=points, intensity=intensity)  # raw pcd input (3,N)
     

        # Project point cloud on binary image and find the valid points (binary_projection)
        height, width = image_size
        height_extended, width_extended = height * self.config.dataset_config.extend_ratio[0], width * self.config.dataset_config.extend_ratio[1]

        *_, rev = self.pt_projection.binary_projection(img_shape=(height_extended, width_extended), intrinsic=intrinsic_matrix, pcd=points.T)
        points = points[rev,:]
        intensity = intensity[rev]
  
        # Subsample the points to keep uniform dimensions 
        points, intensity = self.point_cloud_sampler(points, intensity)

        return points.T, intensity.T

    def __get_depth_image(self, point_cloud: np.ndarray, image_size: Tuple[int, int], intrinsic_matrix: np.ndarray, intensity: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,  torch.Tensor]:
        """
        Generates a depth image and a range image from a given point cloud using projection onto a 2D plane.

        Args:
            point_cloud (np.ndarray): 3D point cloud with shape (3, num_points) containing (x, y, z) coordinates.
            image_size (Tuple[int, int]): Size of the output images in pixels (height, width).
            intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix used for projection.
            intensity (np.ndarray): Intensity values for each point in the point cloud.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Generated depth image, ranges, and intensities.
        """
        # Get size
        height, width = image_size
       

        # Calculate the Euclidean distances (ranges) of each point in the point cloud
        pcd_range = np.linalg.norm(point_cloud, axis=0)
        density = self.get_desnsity(point_cloud=point_cloud.T)
        # intensity = self.scaler_intensity(intensity)
        pcd_range = self.scaler_range(pcd_range)
        density = self.scaler_desnity(density)

        # Perform point cloud projection using provided intrinsic matrix
        u, v, r, rev = self.pt_projection.pcd_projection((height, width), intrinsic_matrix, point_cloud, pcd_range)

        # Create the depth image as a PyTorch tensor
        depth_img = torch.zeros((3, height, width), dtype=torch.float32)
        depth_img[0, v, u] = torch.from_numpy(r).type(torch.float32)
        depth_img[1, v, u] = torch.from_numpy(intensity[rev]).type(torch.float32)
        depth_img[2, v, u] = torch.from_numpy(density[rev]).type(torch.float32)

        # Convert the ranges to a PyTorch tensor
        pcd_range_tensor = torch.from_numpy(pcd_range).type(torch.float32)
        intensity_tensor = torch.from_numpy(intensity).type(torch.float32)
        density_tensor = torch.from_numpy(density).type(torch.float32)

        return depth_img, pcd_range_tensor, intensity_tensor, density_tensor


class KittiDatasetRaw(Dataset):
    """
    Custom dataset class for KITTI raw data, used to load images, point clouds, depth images, and intrinsic/extrinsic camera parameters.
    
    Args:
        config (Config): Configuration object containing dataset settings such as camera ID, dataset path, and transformations.
        split (str): Specifies the dataset split: 'train', 'val', or 'test'. Default is 'train'.
    
    Attributes:
        config (Config): The configuration object containing dataset settings.
        kitti_datalist (list): List of KITTI raw data objects corresponding to the selected sequences.
        sep (list): List of lengths for each sequence in the dataset.
        sumsep (np.ndarray): Cumulative sum of sequence lengths.
        np_to_tensor (ToTensor): Transformation to convert NumPy arrays to PyTorch tensors.
        img_to_tensor (Tf.ToTensor): Transformation to convert images to PyTorch tensors.
        point_cloud_sampler (PointCloudResampler): Sampler to resample point clouds to a fixed number of points.
        point_cloud_filter (PointCloudFilter): Filter to apply to point clouds 
        pt_projection (PointCloudProjection): Projection of point clouds onto a 2D plane.
        get_desnsity (PointCloudDensity): Tool for calculating point cloud density.
        scaler_desnity (MinMaxScaler): Scaler to normalize point cloud density.
        scaler_range (MinMaxScaler): Scaler to normalize point cloud range.
        scaler_intensity (MinMaxScaler): Scaler to normalize point cloud intensity.
    """
   
    def __init__(self, config: Config, split: str = 'train'):
        """
        Initializes the dataset by loading the specified split and checking the sequences.

        Args:
            config (Config): Configuration object containing dataset settings.
            split (str): Specifies the dataset split: 'train', 'val', or 'test'. Default is 'train'.
        """
        self.config = config
        
        # Select the sequences based on the specified split
        if split == 'train':
            sequences = self.config.dataset_config.train_sequences
        elif split == 'val':
            sequences = self.config.dataset_config.val_sequences
        elif split == 'test':
            sequences = self.config.dataset_config.test_sequences

        # Check for meta_data and create it if not present
        dict_len = self.__check_length(sequences)

        # Load KITTI data for each sequence and drive
        self.kitti_datalist = []
        for sequence in sequences:
            for drive in Path(os.path.join(self.config.dataset_path, sequence)).iterdir():
                if drive.is_dir():
                    frames = range(0, dict_len[drive.name.split('/')[-1]])
                    drive = drive.name.split('_')[-2]
                    self.kitti_datalist.append(pykitti.raw(self.config.dataset_path, sequence, drive))
            
        # Compute lengths for each sequence
        self.sep = [len(data) for data in self.kitti_datalist]
        self.sumsep = np.cumsum(self.sep)

        # Initialize transformations and tools
        self.np_to_tensor = ToTensor(tensor_type=torch.float)
        self.img_to_tensor = Tf.ToTensor()
        self.point_cloud_sampler = PointCloudResampler(num_points=self.config.dataset_config.pcd_min_samples)
        self.point_cloud_filter = PointCloudFilter(range_threshold=self.config.dataset_config.max_depth, 
                                                   min_neighbors=2, 
                                                   concat='none')
        self.pt_projection = PointCloudProjection()
        self.get_desnsity = PointCloudDensity(radius=0.3)
        self.scaler_desnity = MinMaxScaler(min_val=0, max_val=self.config.dataset_config.num_neighbours)
        self.scaler_range = MinMaxScaler(min_val=self.config.dataset_config.min_depth, 
                                          max_val=self.config.dataset_config.max_depth)
        self.scaler_intensity = MinMaxScaler(min_val=self.config.dataset_config.min_intensity, 
                                             max_val=self.config.dataset_config.max_intensity)
    

    def __check_length(self, sequences):
        """
        Checks the length of each sequence and the number of Velodyne point cloud files.

        Args:
            sequences (list): List of sequence names to check.

        Returns:
            dict: Dictionary mapping sequence names to the number of Velodyne point cloud files.
        """
        dict_len = dict()
        for sequence in sequences:
            seq_dir = os.path.join(self.config.dataset_path, sequence)
            for drive in Path(seq_dir).iterdir():
                if drive.is_dir():
                    len_velo = len(os.listdir(os.path.join(seq_dir, drive, 'velodyne_points', 'data')))
                    dict_len[drive.name.split('/')[-1]] = len_velo
        return dict_len
    
    def __len__(self):
        """
        Returns the total number of data points in the dataset.

        Returns:
            int: Total number of data points.
        """
        return self.sumsep[-1]

    @staticmethod
    def __check(odom_obj: pykitti.odometry, cam_id: int, seq: str) -> bool:
        """
        Checks the validity of the odometry data for a given sequence and camera.

        Args:
            odom_obj (pykitti.odometry): KITTI odometry object.
            cam_id (int): Camera ID to check for image loading.
            seq (str): Sequence identifier.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        calib = odom_obj.calib
        cam_files_length = len(getattr(odom_obj, 'cam%d_files' % cam_id))
        velo_files_length = len(odom_obj.velo_files)
        head_msg = '[Seq %s]:' % seq
        assert cam_files_length > 0, head_msg + 'None of camera %d files' % cam_id
        assert cam_files_length == velo_files_length, head_msg + "number of cam %d (%d) and velo files (%d) doesn't equal!" % (cam_id, cam_files_length, velo_files_length)
        assert hasattr(calib, 'T_cam0_velo'), head_msg + "Crucial calib attribute 'T_cam0_velo' doesn't exist!"

    def __getitem__(self, index):
        """
        Get a data point from the dataset by index.

        Args:
            index (int): Index of the data point to retrieve.

        Returns:
            dict: Dictionary containing the image, point cloud, range, intensity, density, depth image, intrinsic transformation, and extrinsic transformation.
        """
        group_id = np.digitize(index, self.sumsep, right=False)

        # Retrieve the appropriate dataset and sub-index
        data = self.kitti_datalist[group_id]
        sub_index = index - self.sumsep[group_id - 1] if group_id > 0 else index

        # Retrieve image data
        image, image_size, adjustment_ratio = self.__get_image(data=data, sub_index=sub_index)

        # Get intrinsic and extrinsic parameters
        extrinsic = self.__get_extrinsic(data=data)
        intrinsic_resized, intrinsic_extended = self.__get_intrinsic(data=data, adjustment_ratio=adjustment_ratio)

        # Get point cloud data
        point_cloud, intensity = self.__get_point_cloud(data=data, sub_index=sub_index, extrinsic=extrinsic, 
                                                       image_size=image_size, intrinsic_matrix=intrinsic_extended)

        # Generate depth image and range image
        depth_img, pcd_range_tensor, intensity_tensor, density_tensor = self.__get_depth_image(
            point_cloud=point_cloud, image_size=image_size, intrinsic_matrix=intrinsic_resized, intensity=intensity)

        return dict(img=image,
                    pcd=self.np_to_tensor(point_cloud),
                    pcd_range=pcd_range_tensor,
                    intensity=intensity_tensor,
                    density=density_tensor,
                    depth_img=depth_img,
                    InTran=self.np_to_tensor(intrinsic_resized),
                    ExTran=self.np_to_tensor(extrinsic))

    def __get_extrinsic(self, data):
        """
        Retrieves the extrinsic camera-to-velo transformation matrix.

        Args:
            data (pykitti.raw): KITTI raw data object.

        Returns:
            np.ndarray: 4x4 extrinsic transformation matrix.
        """
        T_cam2velo = getattr(data.calib, 'T_cam%d_velo' % self.config.dataset_config.cam_id)
        return T_cam2velo
    
    def __get_intrinsic(self, data, adjustment_ratio) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the intrinsic camera matrix and extended camera matrix.

        Args:
            data (pykitti.raw): KITTI raw data object.
            adjustment_ratio (np.ndarray): Ratio for adjusting the image size.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Intrinsic matrix and extended intrinsic matrix.
        """
        K_cam = adjustment_ratio @ getattr(data.calib, 'K_cam%d' % self.config.dataset_config.cam_id)
        K_cam = np.diag([self.config.dataset_config.resize_ratio[1], self.config.dataset_config.resize_ratio[0], 1]) @ K_cam
        # Gt extended intrinsic 
        K_cam_extend = K_cam.copy()
        K_cam_extend[0, -1] *= self.config.dataset_config.extend_ratio[0]
        K_cam_extend[1, -1] *= self.config.dataset_config.extend_ratio[1]
        return K_cam, K_cam_extend
    
    def __get_image(self, data, sub_index: int) -> Tuple[torch.Tensor, tuple, np.ndarray]:
        """
        Retrieves an image from the dataset, resizes it, and returns the adjusted image and size.

        Args:
            data (pykitti.raw): KITTI raw data object.
            sub_index (int): Index for the specific frame in the dataset.

        Returns:
            Tuple[torch.Tensor, tuple, np.ndarray]: Image tensor, image size, and adjustment ratio.
        """
        # Load the image
        img, _ = data.get_cam(self.config.dataset_config.cam_id)[sub_index]
        img = self.img_to_tensor(img)
        image_size = img.shape[1:]  # [H, W]

        # Adjust image based on resize configuration
        adjustment_ratio = np.array([img.shape[2] / self.config.dataset_config.input_size[1], 
                                     img.shape[1] / self.config.dataset_config.input_size[0]])
        img = Tf.resize(img, self.config.dataset_config.input_size)
        return img, image_size, adjustment_ratio

    def __get_point_cloud(self, data, sub_index: int, extrinsic: np.ndarray, image_size: tuple, 
                          intrinsic_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves and processes the point cloud data for the specified image frame.

        Args:
            data (pykitti.raw): KITTI raw data object.
            sub_index (int): Index for the specific frame in the dataset.
            extrinsic (np.ndarray): Extrinsic transformation matrix.
            image_size (tuple): Image size.
            intrinsic_matrix (np.ndarray): Intrinsic camera matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed point cloud and intensity values.
        """
        velo = data.get_velo(sub_index) 
        point_cloud = self.point_cloud_filter(velo, extrinsic)
        resampled_pc = self.point_cloud_sampler(point_cloud)
        # Point cloud intensity
        intensity = self.scaler_intensity(resampled_pc[:, 3])
        return resampled_pc[:, :3].T, intensity.T
   
    def __get_depth_image(self, point_cloud: np.ndarray, image_size: Tuple[int, int], intrinsic_matrix: np.ndarray, intensity: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,  torch.Tensor]:
        """
        Generates a depth image and a range image from a given point cloud using projection onto a 2D plane.

        Args:
            point_cloud (np.ndarray): 3D point cloud with shape (3, num_points) containing (x, y, z) coordinates.
            image_size (Tuple[int, int]): Size of the output images in pixels (height, width).
            intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix used for projection.
            intensity (np.ndarray): Intensity values for each point in the point cloud.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Generated depth image, ranges, and intensities.
        """
        # Get size
        height, width = image_size
       

        # Calculate the Euclidean distances (ranges) of each point in the point cloud
        pcd_range = np.linalg.norm(point_cloud, axis=0)
        density = self.get_desnsity(point_cloud=point_cloud.T)
        # intensity = self.scaler_intensity(intensity)
        pcd_range = self.scaler_range(pcd_range)
        density = self.scaler_desnity(density)

        # Perform point cloud projection using provided intrinsic matrix
        u, v, r, rev = self.pt_projection.pcd_projection((height, width), intrinsic_matrix, point_cloud, pcd_range)

        # Create the depth image as a PyTorch tensor
        depth_img = torch.zeros((3, height, width), dtype=torch.float32)
        depth_img[0, v, u] = torch.from_numpy(r).type(torch.float32)
        depth_img[1, v, u] = torch.from_numpy(intensity[rev]).type(torch.float32)
        depth_img[2, v, u] = torch.from_numpy(density[rev]).type(torch.float32)

        # Convert the ranges to a PyTorch tensor
        pcd_range_tensor = torch.from_numpy(pcd_range).type(torch.float32)
        intensity_tensor = torch.from_numpy(intensity).type(torch.float32)
        density_tensor = torch.from_numpy(density).type(torch.float32)

        return depth_img, pcd_range_tensor, intensity_tensor, density_tensor