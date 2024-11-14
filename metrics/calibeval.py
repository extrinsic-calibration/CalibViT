
from config import Config
import os
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_euler_angles
import numpy as np
from transform import SO3
import json 
from typing import Optional

class CalibEval(object):
    def __init__(self, config: Config, translation_threshold: Optional[float]=None, rotation_threshold: Optional[float]=None) -> None:
        """
        Initialize the CalibEval object.

        Parameters:
            config (Config): Configuration object containing evaluation parameters.
            translation_threshold (float): Threshold for translation error to consider as successful calibration.
            rotation_threshold (float): Threshold for rotation error (in degrees) to consider as successful calibration.
        """
        self.config = config
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.so3 = SO3()  # Initialize SO3 for handling rotation transformations
        self.loss_r = []   # List to store rotation losses
        self.loss_t = []   # List to store translation losses
        self.geodesic = [] # List to store geodesic distances
        self.success_idx = [] # List to store indices of successful calibrations
        self.pred = []     # List to store predicted values
        self.reset()       # Reset the lists for a fresh start

    def reset(self) -> None:
        """
        Reset all stored losses and calibration results.
        """
        self.loss_r = []
        self.loss_t = []
        self.geodesic = []
        self.success_idx = []
        self.pred = []

    def add_batch(self, gt_tf: torch.Tensor, pred_tf: torch.Tensor, idx: Optional[int] = None, return_results: bool = False) -> None:
        """
        Add a batch of ground truth and predicted transformations.

        Parameters:
            gt_tf (torch.Tensor): Ground truth transformation tensors.
            pred_tf (torch.Tensor): Predicted transformation tensors.
            idx (Optional[int]): Index of the current batch (used for tracking successful calibrations).
        """
        # Get error between predicted and ground truth transformations
        error = pred_tf.bmm(gt_tf)

        # Get ground truth and predicted rotation and translation vectors
        gt_rot, gt_trans = self.get_rotation_translation_from_transform(gt_tf.cpu())
        pred_rot, pred_trans = self.get_rotation_translation_from_transform(pred_tf.cpu())

        # Convert rotation matrices to Euler angles
        gt_euler_angles = self.rotation_matrix_to_euler(gt_rot)
        pred_euler_angles = self.rotation_matrix_to_euler(pred_rot)

        # Calculate errors in rotation and translation
        err_rot, err_trans = self.get_rotation_translation_from_transform(error.cpu())
        err_euler_angles = self.rotation_matrix_to_euler(err_rot)

        # Geodesic error 
        geodesic_distance = self.geodesic_distance(error)

        if return_results:
            return err_rot, err_trans, err_euler_angles, geodesic_distance

        # Store losses and errors
        self.loss_r.extend(err_euler_angles.tolist())
        self.loss_t.extend(err_trans.tolist())
        self.pred.extend(torch.cat((pred_euler_angles, pred_trans), dim=1).tolist())
        self.geodesic.append(self.geodesic_distance(error))

        # Check if the current batch is successful based on thresholds
        if self.translation_threshold is not None and self.rotation_threshold is not None:
            if idx is not None:
                for i in range(self.config.world_size):
                    if np.mean(np.abs(err_trans.tolist())) < self.translation_threshold and np.mean(np.abs(err_euler_angles.tolist())) < self.rotation_threshold:
                        self.success_idx.append(idx)

    def get_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get statistics of the calibration errors.

        Returns:
            tuple: A tuple containing:
                - loss_r (np.ndarray): Mean rotation loss for successful calibrations.
                - loss_t (np.ndarray): Mean translation loss for successful calibrations.
                - geodesic (np.ndarray): Mean geodesic distance for successful calibrations.
        """
        if self.success_idx:
            loss_r = np.abs(np.asarray(self.loss_r)[np.asarray(self.success_idx)]).mean(axis=0)
            loss_t = np.abs(np.asarray(self.loss_t)[np.asarray(self.success_idx)]).mean(axis=0)
            geodesic = np.asarray(self.geodesic)[np.asarray(self.success_idx)].mean(axis=0)
        else:
            loss_r = np.abs(np.asarray(self.loss_r)).mean(axis=0)
            loss_t = np.abs(np.asarray(self.loss_t)).mean(axis=0)
            geodesic = np.asarray(self.geodesic).mean(axis=0)
        return loss_r, loss_t, geodesic

    def getSD(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get standard deviation of the calibration errors.

        Returns:
            tuple: A tuple containing:
                - loss_r (np.ndarray): Standard deviation of rotation loss.
                - loss_t (np.ndarray): Standard deviation of translation loss.
        """
        if self.success_idx:
            loss_r = np.abs(np.asarray(self.loss_r)[np.asarray(self.success_idx)]).std(axis=0)
            loss_t = np.abs(np.asarray(self.loss_t)[np.asarray(self.success_idx)]).std(axis=0)
        else:
            loss_r = np.abs(np.asarray(self.loss_r)).std(axis=0)
            loss_t = np.abs(np.asarray(self.loss_t)).std(axis=0)
        return loss_r, loss_t
    
    def compute_recall(self) -> float:
        """
        Compute the recall of the calibration results.

        Returns:
            float: The recall value as a ratio of successful calibrations to total attempts.
        """
        if self.loss_r:
            return len(self.success_idx) / len(self.loss_r)
        else:
            return 0.0


    def geodesic_distance(self,x:torch.Tensor,)->list:
        """
        geodesic distance for evaluation

        Args:
            x (torch.Tensor): (B,4,4)
        Returns:
            torch.Tensor(1),torch.Tensor(1): distance of component R and T 
        
        """
        # Compute the relative rotation matrix
        R_error = x[:, :3, :3]    # Shape: (B, 3, 3)
        
        # Compute the geodesic error (rotation angle)
        trace_R_error = torch.diagonal(R_error, dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace_R_error - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp to avoid numerical issues
        theta = torch.acos(cos_theta)  # Shape: (B,)
        
        # Compute the translation error (Euclidean distance)
        translation_error = torch.norm(x[:, :3, 3], dim=1)  # Shape: (B,)
        
        # Combine the rotation and translation errors
        return [theta.mean().item(), translation_error.mean().item()] # Shape: (B,)
    

    @staticmethod
    def rotation_matrix_to_euler(rotation_matrix):
        # Ensure rotation_matrix is a torch.Tensor
        assert isinstance(rotation_matrix, torch.Tensor), "Input must be a PyTorch tensor"
        
        # Convert rotation matrices to Euler angles
        euler_angles = matrix_to_euler_angles(rotation_matrix, convention="XYZ") #, "Y", "X")
        euler_angles = (torch.rad2deg(euler_angles))
        return euler_angles

    @staticmethod
    def get_rotation_translation_from_transform(tf):
        r"""Decompose transformation matrix into rotation matrix and translation vector.

        Args:
            tf (Tensor): (*, 4, 4)

        Returns:
            rotation (Tensor): (*, 3, 3)
            translation (Tensor): (*, 3)
        """
        rotation = tf[..., :3, :3]
        translation = tf[..., :3, 3]
        return rotation, translation

    @staticmethod
    def compute_norm(tensor_a, tensor_b):
        """
        Compute the L2 norm (Euclidean distance) between corresponding elements of two tensors.

        Args:
            tensor_a (torch.Tensor): Tensor of shape [B, 3]
            tensor_b (torch.Tensor): Tensor of shape [B, 3]

        Returns:
            torch.Tensor: Tensor of shape [B, 3] containing the L2 norm between corresponding elements.
        """
        # Ensure that both tensors have the same shape
        assert tensor_a.shape == tensor_b.shape, "Shapes of input tensors must be the same"

        # Compute L2 norm between corresponding elements
        norm = tensor_a - tensor_b

        return norm


    @staticmethod
    def relative_rotation_error(gt_rotations, rotations):
        r"""Isotropic Relative Rotation Error.

        RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

        Args:
            gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
            rotations (Tensor): estimated rotation matrix (*, 3, 3)

        Returns:
            rre (Tensor): relative rotation errors (*)
        """
        mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
        trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
        x = 0.5 * (trace - 1.0)
        x = x.clamp(min=-1.0, max=1.0)
        x = torch.arccos(x)
        rre = 180.0 * x / np.pi
        return rre

    
    def save_results(self):
        """
        Save the calibration results to a JSON file.

        This function gathers the statistical results of the calibration process,
        including the mean and standard deviation of the errors for translation 
        and rotation. It then organizes these results into a dictionary and saves 
        it as a JSON file. The file name is constructed based on various parameters 
        such as dataset name, model configuration, and iteration count.

        The output JSON file will contain:
            - dataset: Name of the dataset used.
            - model: Model name and version used.
            - translation: Maximum translation values from the dataset configuration.
            - rotation: Maximum rotation degrees from the dataset configuration.
            - distribution: Distribution type used in the dataset configuration.
            - iter: Number of inner iterations for the model.
            - pred_calib: Predicted calibration values.
            - error_calib: Concatenated loss values for rotation and translation errors.
            - mean_error: Mean error values for rotation, translation, and ground truth.
            - sd: Standard deviation of the rotation and translation errors.
            - mean_sd: Mean of the standard deviation values for rotation and translation.
            - success_idx: Indices indicating successful calibration.
            - recall: Recall metric calculated from the results.
        """
        
        # Get mean values for rotation, translation, and ground truth calibration
        r, t, g = self.get_stats()

        # Get standard deviations for rotation and translation errors
        sd_t, sd_r = self.getSD()
        
        # Organize results into a dictionary
        results = { 
            "dataset": self.config.dataset,
            "model": self.config.model_config.model + self.config.model_config.version if self.config.model_config.version else self.config.model_config.model,
            'translation': self.config.dataset_config.max_tran,
            'rotation': self.config.dataset_config.max_deg,
            'distribution': self.config.dataset_config.distribution,
            'iter': self.config.model_config.inner_iter,
            'pred_calib': self.pred,
            'error_calib': np.concatenate((self.loss_r, self.loss_t), axis=1).tolist(),
            "mean_error": sum([r.tolist(), t.tolist(), g.tolist()], []),   
            "sd": sum([sd_r.tolist(), sd_t.tolist()], []),
            "mean_sd": [np.mean(sd_r).tolist(), np.mean(sd_t).tolist()],
            'success_idx': self.success_idx,
            "recall": self.compute_recall()
        }

        # Create the output JSON filename based on configuration parameters
        json_filename = os.path.join(
            self.config.prediction_path, 
            "results_" + self.config.mode + '_' + self.config.dataset + '_' + 
            self.config.dataset_config.distribution + '_' + str(self.config.dataset_config.max_deg) + '_' + 
            str(self.config.dataset_config.max_tran) + '_' + str(self.config.model_config.inner_iter) + '_' +  
            (self.config.model_config.model + '_' + self.config.model_config.version if self.config.model_config.version else self.config.model_config.model) + 
            ".json"
        )
        
        # Save the results to a JSON file
        with open(json_filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)
