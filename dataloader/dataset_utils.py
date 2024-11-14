import torch
import open3d as o3d
import numpy as np
from typing import Tuple, Optional
from scipy.spatial import KDTree

import numpy as np
import torch
from typing import Tuple

class Resampler:
    """
    Resamples a given point cloud `[N, D]` to `[M, D]`.

    Used for training to handle point clouds with a variable number of points.

    Args:
        num (int): The number of points to sample (M). If `num < 0`, returns a random permutation 
                   of all points. If `num > N`, oversamples by repeating points.

    """
    def __init__(self, num: int):
        self.num = num

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Resample the input point cloud.

        Args:
            x (np.ndarray): Input point cloud of shape `[N, D]`.

        Returns:
            np.ndarray: Resampled point cloud of shape `[M, D]`.
        """
        num_points = x.shape[0]
        idx = np.random.permutation(num_points)
        if self.num < 0:
            return x[idx]  # Return a random permutation of all points.
        elif self.num <= num_points:
            idx = idx[:self.num]  # Sample exactly 'self.num' points.
            return x[idx]
        else:
            # Oversample by repeating points.
            idx = np.hstack([idx, np.random.choice(num_points, self.num - num_points, replace=True)])
            return x[idx]


class MaxResampler:
    """
    Resamples a given point cloud `[N, D]` to `[M, D]` where `M <= num`.

    Used for testing to handle point clouds with a fixed maximum number of points.

    Args:
        num (int): The maximum number of points to sample (M).
        seed (int, optional): Seed for reproducible random sampling. Defaults to 8080.
    """
    def __init__(self, num: int, seed: int = 8080):
        self.num = num
        np.random.seed(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Resample the input point cloud.

        Args:
            x (np.ndarray): Input point cloud of shape `[N, D]`.

        Returns:
            np.ndarray: Resampled point cloud of shape `[M, D]`, where `M <= num`.
        """
        num_points = x.shape[0]
        x_ = np.random.permutation(x)
        if num_points <= self.num:
            return x_  # Return all points in random order.
        else:
            return x_[:self.num]  # Return the first 'num' points.


class ToTensor:
    """
    Converts a NumPy array to a PyTorch tensor with a specified data type.

    Args:
        tensor_type (torch.dtype, optional): The desired tensor data type. Defaults to `torch.float`.
    """
    def __init__(self, tensor_type: torch.dtype = torch.float):
        self._tensor_type = tensor_type

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """
        Convert the input NumPy array to a PyTorch tensor.

        Args:
            x (np.ndarray): Input NumPy array.

        Returns:
            torch.Tensor: Converted tensor.
        """
        return torch.from_numpy(x).type(self._tensor_type)


class PointCloudProjection:
    """
    Contains static methods for projecting 3D point clouds into 2D images.
    """

    @staticmethod
    def pcd_projection(img_shape: Tuple[int, int], intrinsic: np.ndarray, pcd: np.ndarray, range_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Project a 3D point cloud onto a 2D depth image.

        Args:
            img_shape (Tuple[int, int]): Image dimensions `(H, W)`.
            intrinsic (np.ndarray): Camera intrinsic matrix of shape `[3, 3]`.
            pcd (np.ndarray): 3D points array of shape `[3, N]`.
            range_arr (np.ndarray): 1D array of ranges for points, shape `[N]`.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - Projected u-coordinates `[M]`.
                - Projected v-coordinates `[M]`.
                - Ranges `[M]` for valid points.
                - Boolean mask `[N]` indicating valid projections.
        """
        H, W = img_shape
        proj_pcd = intrinsic @ pcd
        u, v, w = proj_pcd[0, :], proj_pcd[1, :], proj_pcd[2, :]

        u = (u / w).astype(np.int64)
        v = (v / w).astype(np.int64)

        rev = (0 <= u) & (u < W) & (0 <= v) & (v < H) & (w > 0)

        u = u[rev]
        v = v[rev]
        r = range_arr[rev]

        return u, v, r, rev

    @staticmethod
    def binary_projection(img_shape: Tuple[int, int], intrinsic: np.ndarray, pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project a 3D point cloud onto a 2D binary image.

        Args:
            img_shape (Tuple[int, int]): Image dimensions `(H, W)`.
            intrinsic (np.ndarray): Camera intrinsic matrix of shape `[3, 3]`.
            pcd (np.ndarray): 3D points array of shape `[3, N]`.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Projected u-coordinates `[M]`.
                - Projected v-coordinates `[M]`.
                - Boolean mask `[N]` indicating valid projections.
        """
        H, W = img_shape
        proj_pcd = intrinsic @ pcd

        u, v, w = proj_pcd[0, :], proj_pcd[1, :], proj_pcd[2, :]

        u = (u / w).astype(np.int64)
        v = (v / w).astype(np.int64)

        rev = (0 <= u) & (u < W) & (0 <= v) & (v < H) & (w > 0)

        return u, v, rev


class PointCloudFilter:
    """PointCloudFilter class for processing point cloud data.

    Args:
        concat (str, optional): Concatenation operation for normal estimation.
            Possible values: 'none', 'xyz', or 'zero-mean'. Defaults to 'none'.
        range_threshold (float, optional): Threshold for filtering points based on range. Defaults to 100.0.
        radius (float, optional): Radius for neighbor search. Defaults to 0.4.
        min_neighbors (int, optional): Minimum number of neighbors to be considered an inlier. Defaults to 2.
    """
    def __init__(self, concat: str = 'none', range_threshold: float = 100.0, radius: float = 0.3, min_neighbors: int = 2):
        self._concat = concat
        self._range_threshold = range_threshold
        self._radius = radius
        self._min_neighbors = min_neighbors

    def _filter_points_by_range(self, points: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter points based on the specified range threshold.

        Args:
            points (np.ndarray): Input point cloud data as a NumPy array (N, 3).
            intensity (np.ndarray): Intensity values corresponding to each point.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered point cloud data and intensity array.
        """
        ranges = np.linalg.norm(points, axis=1)
        idx = ranges <= self._range_threshold
        filtered_points = points[idx]
        filtered_intensity = intensity[idx]
        return filtered_points, filtered_intensity

    def _remove_outliers(self, points: np.ndarray) -> np.ndarray:
        """
        Remove outlier points that have fewer neighbors than the specified threshold.

        Args:
            points (np.ndarray): Input point cloud data as a NumPy array (N, 3).

        Returns:
            np.ndarray: Point cloud data with outliers removed.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        cl, ind = pcd.remove_radius_outlier(nb_points=self._min_neighbors, radius=self._radius)
        filtered_points = np.array(pcd.select_by_index(ind).points, dtype=np.float32)
        return filtered_points

    def __call__(self, points: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the input point cloud data and return the result.

        Args:
            points (np.ndarray): Input point cloud data as a NumPy array (N, 3).
            intensity (np.ndarray): Intensity values corresponding to each point.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed point cloud data and intensity array.
                The output shape depends on the 'concat' mode:
                - 'none': (N, 3)
                - 'xyz': (N, 6)
                - 'zero-mean': (N, 6)
        """
        filtered_points, filtered_intensity = self._filter_points_by_range(points, intensity)
        return filtered_points, filtered_intensity


class PointCloudResampler:
    """PointCloudResampler class for resampling point clouds.

    Args:
        num_points (int, optional): The desired number of points in the resampled point cloud. Defaults to 1024.
    """
    def __init__(self, num_points: int = 1024):
        self._num_points = num_points

    def __call__(self, point_cloud: np.ndarray, intensity: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Resample the input point cloud to the desired number of points.

        Args:
            point_cloud (np.ndarray): The input point cloud as a numpy array, where each row represents a point.
            intensity (Optional[np.ndarray]): The intensity values associated with each point in the point cloud.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: The resampled point cloud and the corresponding intensity values.
        """
        num_points_in_cloud = point_cloud.shape[0]
       
        if self._num_points != -1:
            if num_points_in_cloud <= self._num_points:
                pad_count = self._num_points - num_points_in_cloud
                pad_indices = np.random.choice(num_points_in_cloud, pad_count, replace=True)
                padded_cloud = np.vstack((point_cloud, point_cloud[pad_indices]))
                if intensity is not None:
                    padded_intensity = np.hstack((intensity, intensity[pad_indices]))
                    return padded_cloud, padded_intensity
                return padded_cloud, None
            else:
                selected_indices = np.random.choice(num_points_in_cloud, self._num_points, replace=False)
                resampled_cloud = point_cloud[selected_indices]
                if intensity is not None:
                    resampled_intensity = intensity[selected_indices]
                    return resampled_cloud, resampled_intensity
                return resampled_cloud, None
        else:
            if intensity is not None:
                return point_cloud, intensity
            return point_cloud, None


class MinMaxScaler:
    """MinMaxScaler class for scaling data to the range [0, 1].

    Args:
        min_val (float): Minimum value of the data range to be scaled (default: 0).
        max_val (float): Maximum value of the data range to be scaled (default: 100).
    """
    def __init__(self, min_val: float = 0, max_val: float = 100):
        self.min_val = min_val
        self.max_val = max_val
        self.scale_ = 1 / (self.max_val - self.min_val)
        self.min_ = -self.min_val * self.scale_

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the data to the range [0, 1].

        Args:
            data (np.ndarray): Data to be scaled.

        Returns:
            np.ndarray: Scaled data in the range [0, 1].
        """
        return (data * self.scale_) + self.min_

    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data from the range [0, 1] back to the original range.

        Args:
            data (np.ndarray): Data to be inverse transformed.

        Returns:
            np.ndarray: Data in the original range.
        """
        return (data - self.min_) / self.scale_ + self.min_val


class PointCloudDensity:
    """PointCloudDensity class for computing the density of points in a point cloud.

    Args:
        radius (float): Radius within which to compute density.
    """
    def __init__(self, radius: float) -> None:
        self.radius = radius
        self.densities = None

    def __call__(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Compute the density of points within the specified radius for each point in the point cloud.

        Args:
            point_cloud (np.ndarray): Input point cloud data as a NumPy array (N, 3).

        Returns:
            np.ndarray: Array of densities for each point in the point cloud.
        """
        tree = KDTree(point_cloud)
        densities = np.zeros(point_cloud.shape[0])
        for i, point in enumerate(point_cloud):
            indices = tree.query_ball_point(point, self.radius, p=2)
            densities[i] = len(indices)
        return densities
