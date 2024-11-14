# torch 
import torch
from torch.utils.data.dataset import Dataset 

# Tools
from transform import UniformTransformSE3
from transform import SE3

# extras 
from config import Config
import numpy as np 


class Perturbation(Dataset):
    def __init__(self, config: Config, dataset: Dataset):
        """
        Initialize the Perturbation class to apply perturbations to the point cloud and depth images.

        Args:
            config (Config): Configuration object containing dataset and perturbation settings.
            dataset (Dataset): The torch dataset containing the point cloud and image data.
        """
        # Config 
        self.config = config

        # Ensure pooling size is odd for constant image size
        assert (self.config.dataset_config.pooling_size-1) % 2 == 0, 'pooling size must be odd to keep image size constant'
        self.pooling = torch.nn.MaxPool2d(kernel_size=self.config.dataset_config.pooling_size, stride=1,
                                           padding=(self.config.dataset_config.pooling_size-1)//2)
        
        # Dataset initialization
        self.dataset = dataset

        # Input decalibration file setup
        self.file = None
        if self.config.mode == 'val':
            self.file = self.config.dataset_config.val_perturb_file
        elif self.config.mode == 'test':
            self.file = self.config.dataset_config.test_perturb_file

        # Read perturbation file or generate transformations
        if self.file is not None:
            self.perturb = torch.from_numpy(np.loadtxt(self.file, dtype=np.float32, delimiter=','))[None, ...]  # (1, N, 6)
            print(">> Using:{}".format(self.file))
        else:
            # Create random transformation generators for different types of noise (uniform, gaussian, inverse Gaussian)
            self.transform_uniform = UniformTransformSE3(max_deg=self.config.dataset_config.max_deg,
                                                        max_tran=self.config.dataset_config.max_tran,
                                                        distribution='uniform',
                                                        mag_randomly=self.config.dataset_config.mag_randomly)
                                                        
            self.transform_gaussian = UniformTransformSE3(max_deg=self.config.dataset_config.max_deg,
                                                         max_tran=self.config.dataset_config.max_tran,
                                                         distribution='gaussian',
                                                         mag_randomly=self.config.dataset_config.mag_randomly)

            self.transform_inv_gaussian = UniformTransformSE3(max_deg=self.config.dataset_config.max_deg,
                                                              max_tran=self.config.dataset_config.max_tran,
                                                              distribution='inverse_gaussian',
                                                              mag_randomly=self.config.dataset_config.mag_randomly)
        
        # SE3 transformation tools for coordinate transformations
        self.se3 = SE3()

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> dict:
        """Get a perturbed sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: Perturbed sample containing uncalibrated point cloud, depth image, and igt (SE3 transform).
        """
        # Retrieve original data sample
        data = self.dataset[index]
        H, W = data['img'].shape[-2:]  # Image dimensions (height, width)
        calibed_pcd = data['pcd']  # Calibrated point cloud (3, N)
        InTran = data['InTran']  # Intrinsic transformation matrix (3, 3)

        # If no perturbation file, randomly generate perturbation
        if self.file is None:
            random_num = np.random.randint(3)
            if random_num == 0:
                randm_transfrom = self.transform_uniform
            elif random_num == 1:
                randm_transfrom = self.transform_gaussian
            else:
                randm_transfrom = self.transform_inv_gaussian

            # Apply the random transformation to the point cloud
            _uncalibed_pcd = randm_transfrom(calibed_pcd[None, :, :]).squeeze(0)  # (3, N)
            igt = randm_transfrom.igt.squeeze(0)  # (4, 4)
        else:
            # Apply predefined perturbation (read from file)
            igt = self.se3.exp(self.perturb[:, index, :])  # (1, 6) -> (1, 4, 4)
            _uncalibed_pcd = self.se3.transform(igt, calibed_pcd[None, ...]).squeeze(0)  # (3, N)
            igt.squeeze_(0)  # (4, 4)

        # Initialize an empty depth image for storing the uncalibrated point cloud data
        _uncalibed_depth_img = torch.zeros_like(data['depth_img'], dtype=torch.float32)

        # Project the uncalibrated point cloud to the image plane
        proj_pcd = InTran.matmul(_uncalibed_pcd)  # (3, 3) x (3, N) -> (3, N)
        proj_x = (proj_pcd[0, :] / proj_pcd[2, :]).type(torch.long)
        proj_y = (proj_pcd[1, :] / proj_pcd[2, :]).type(torch.long)
        rev = (0 <= proj_x) * (proj_x < W) * (0 <= proj_y) * (proj_y < H) * (proj_pcd[2, :] > 0)
        proj_x = proj_x[rev]
        proj_y = proj_y[rev]

        # Populate the depth image with the uncalibrated point cloud data
        _uncalibed_depth_img[0, proj_y, proj_x] = data['pcd_range'][rev]  # Range values
        _uncalibed_depth_img[1, proj_y, proj_x] = data['intensity'][rev]  # Intensity values
        _uncalibed_depth_img[2, proj_y, proj_x] = data['density'][rev]  # Density values

        # Prepare new data dictionary to return
        new_data = dict(uncalibed_pcd=_uncalibed_pcd, uncalibed_depth_img=_uncalibed_depth_img, igt=igt.float())
        data.update(new_data)

        # Apply pooling to depth images (original and uncalibrated)
        data['depth_img'] = self.pooling(data['depth_img'][None, ...]).squeeze(0)
        data['uncalibed_depth_img'] = self.pooling(data['uncalibed_depth_img'][None, ...]).squeeze(0)

        return data
