
# general formating
import os 
import time
from typing import Tuple, Union, Optional
from logger import Recorder
from config import Config

# Torch 
import torch 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# tools 
import numpy as np 
from transform import UniformTransformSE3
import tools 

# Modules
from .kitty_dataset import KittiDataset, KittiDatasetRaw
from .nuscenes_dataset import NuScenesLoader, NuScenesDataset
from .perturbation import Perturbation

def load_dataset(config: Config, recorder: Optional[Recorder]) -> Union[
        Tuple[DataLoader, Optional[DistributedSampler]],
        Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]],
        Tuple[Optional[DataLoader], DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]
    ]:
    """
    Loads the appropriate dataset based on the configuration and prepares DataLoaders.

    Args:
        config (Config): Configuration object containing dataset and model settings.
        recorder (Optional[Recorder]): Recorder object for logging messages, can be None.

    Returns:
        Union[Tuple[DataLoader, Optional[DistributedSampler]], 
              Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]], 
              Tuple[None, DataLoader, None, Optional[DistributedSampler]]]:
            Depending on the dataset, mode, and distributed setup, returns the appropriate DataLoaders and Samplers.
    """
    if config.dataset == 'nuscenes':
        return __nuscenes(config=config, recorder=recorder)
    elif config.dataset == 'kitti':
        return __kitti(config=config, recorder=recorder)
    elif config.dataset == 'kittiRaw':
        return __kitti_raw(config=config, recorder=recorder)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")


def __nuscenes(config: Config, recorder: Optional[Recorder], limscenes: int = -1) -> Union[
        Tuple[DataLoader, Optional[DistributedSampler]],
        Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]],
        Tuple[None, DataLoader, None, Optional[DistributedSampler]]
    ]:
    """
    Loads the NuScenes dataset and prepares DataLoaders for training, validation, or testing.

    Args:
        config (Config): Configuration object containing dataset and model settings.
        recorder (Optional[Recorder]): Recorder object for logging messages, can be None.
        limscenes (int): Limit the number of scenes to load (default: -1, load all scenes).

    Returns:
        Union[Tuple[DataLoader, Optional[DistributedSampler]], 
              Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]], 
              Tuple[None, DataLoader, None, Optional[DistributedSampler]]]:
            Depending on the mode and distributed setup, returns the appropriate DataLoaders and Samplers.
            - In "train" mode, returns training and validation DataLoaders and their corresponding Samplers.
            - In "test" mode, returns only the test DataLoader and Sampler.
    """
    
    loader = NuScenesLoader()

    # Load NuScenes dataset based on mode (train, val, test)
    if config.mode == 'test':
        # Load the test dataset
        nusc = loader(data_root=config.dataset_path, version='v1.0-test', verbose=False)
        test_set = NuScenesDataset(nusc=nusc, config=config, split='test', limscenes=None)
        test_length = len(test_set)

        # Check and create perturbation file for validation if necessary
        if not os.path.exists(config.dataset_config.test_perturb_file):
            if recorder is not None:
                recorder.log_message(">> Validation perturb file doesn't exist, creating one.", level='info')
            __create_perturb_file(config=config, length=test_length, path_to_file=config.dataset_config.test_perturb_file)
        else:  
            test_seq = np.loadtxt(config.dataset_config.test_perturb_file, delimiter=',')
            if test_length != test_seq.shape[0]:
                if recorder is not None:
                    recorder.log_message(f'>> Incompatible validation length {test_length}!={test_seq.shape[0]}', level='warning')
                __create_perturb_file(config=config, length=test_length, path_to_file=config.dataset_config.test_perturb_file)
                if recorder is not None:
                    recorder.log_message('>> Validation perturb file rewritten.', level='info')

        test_set = Perturbation(dataset=test_set, config=config)
        
        test_drop_last = len(test_set) % config.model_config.batch_size_val == 1

        if tools.is_dist_avail_and_initialized():
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)
            test_loader = DataLoader(
                test_set,
                batch_size=config.model_config.batch_size_val,
                num_workers=config.num_workers,
                drop_last=test_drop_last,
                sampler=test_sampler,
                pin_memory=True,
                prefetch_factor=2,
            )
            return test_loader, test_sampler
        else:
            test_loader = DataLoader(
                test_set,
                batch_size=config.model_config.batch_size_val,
                num_workers=config.num_workers,
                shuffle=False,
                drop_last=test_drop_last,
                pin_memory=True,
                prefetch_factor=2,
            )
            return test_loader, None

    # Load the training and validation dataset for other modes
    nusc = loader(data_root=config.dataset_path, version='v1.0-trainval', verbose=False)

    # Prepare training dataset
    if config.mode == 'train':
        train_set = NuScenesDataset(nusc=nusc, config=config, split='train', limscenes=None)
        train_set = Perturbation(dataset=train_set, config=config)
        
        train_drop_last = len(train_set) % config.model_config.batch_size == 1

        if tools.is_dist_avail_and_initialized():
            train_sampler = DistributedSampler(train_set)
            train_loader = DataLoader(
                train_set,
                batch_size=config.model_config.batch_size,
                num_workers=config.num_workers,
                drop_last=train_drop_last,
                sampler=train_sampler,
                pin_memory=True,
                prefetch_factor=2,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=config.model_config.batch_size,
                num_workers=config.num_workers,
                shuffle=True,
                drop_last=train_drop_last,
                pin_memory=True,
                prefetch_factor=2,
            )

    # Prepare validation dataset
    val_set = NuScenesDataset(nusc=nusc, config=config, split='val', limscenes=None)
    val_length = len(val_set)

    # Check and create perturbation file for validation if necessary
    if not os.path.exists(config.dataset_config.val_perturb_file):
        if recorder is not None:
            recorder.log_message(">> Validation perturb file doesn't exist, creating one.", level='info')
        __create_perturb_file(config=config, length=val_length, path_to_file=config.dataset_config.val_perturb_file)
    else:  
        val_seq = np.loadtxt(config.dataset_config.val_perturb_file, delimiter=',')
        if val_length != val_seq.shape[0]:
            if recorder is not None:
                recorder.log_message(f'>> Incompatible validation length {val_length}!={val_seq.shape[0]}', level='warning')
            __create_perturb_file(config=config, length=val_length, path_to_file=config.dataset_config.val_perturb_file)
            if recorder is not None:
                recorder.log_message('>> Validation perturb file rewritten.', level='info')

    val_set = Perturbation(dataset=val_set, config=config)
    
    val_drop_last = len(val_set) % config.model_config.batch_size_val == 1

    # Prepare DataLoader for validation
    if tools.is_dist_avail_and_initialized():
        val_sampler = DistributedSampler(val_set, shuffle=False)
        val_loader = DataLoader(
            val_set,
            batch_size=config.model_config.batch_size_val,
            num_workers=config.num_workers,
            drop_last=val_drop_last,
            sampler=val_sampler,
            pin_memory=True,
            prefetch_factor=2,
        )
    else:
        val_loader = DataLoader(
            val_set,
            batch_size=config.model_config.batch_size_val,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=val_drop_last,
            pin_memory=True,
            prefetch_factor=2,
        )

    if recorder is not None:
        recorder.log_message(">> Data loaded successfully.", level='info')

    # Return appropriate DataLoaders based on mode and distributed setup
    if config.mode == "train":
        if tools.is_dist_avail_and_initialized():
            return train_loader, val_loader, train_sampler, val_sampler
        else:
            return train_loader, val_loader, None, None
    else:
        if tools.is_dist_avail_and_initialized():
            return None, val_loader, None, val_sampler
        else:
            return None, val_loader, None, None



def __kitti(config: Config, recorder: Optional[Recorder]) -> Union[
        Tuple[DataLoader, Optional[DistributedSampler]],
        Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]],
        Tuple[None, DataLoader, None, Optional[DistributedSampler]]
    ]:
    """
    Loads the KITTI dataset and prepares DataLoaders for training, validation, or testing.

    Args:
        config (Config): Configuration object containing dataset and model settings.
        recorder (Optional[Recorder]): Recorder object for logging messages, can be None.

    Returns:
        Union[Tuple[DataLoader, Optional[DistributedSampler]], 
              Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]], 
              Tuple[None, DataLoader, None, Optional[DistributedSampler]]]:
            Depending on the mode and distributed setup, returns the appropriate DataLoaders and Samplers.
    """
    # Handling test mode
    if config.mode == 'test':
        # Initialize test dataset
        test_set = KittiDataset(config=config, split='test')
        test_length = len(test_set)

        # Check if perturbation file exists for test
        if not os.path.exists(config.dataset_config.test_perturb_file):
            if recorder is not None:
                recorder.log_message(">> Validation perturb file doesn't exist, creating one.", level='info')
            __create_perturb_file(config=config, length=test_length, path_to_file=config.dataset_config.test_perturb_file)
        else:
            test_seq = np.loadtxt(config.dataset_config.test_perturb_file, delimiter=',')
            if test_length != test_seq.shape[0]:
                if recorder is not None:
                    recorder.log_message(f'>> Incompatible validation length {test_length}!={test_seq.shape[0]}', level='warning')
                __create_perturb_file(config=config, length=test_length, path_to_file=config.dataset_config.test_perturb_file)
                if recorder is not None:
                    recorder.log_message('>> Validation perturb file rewritten.', level='info')

        # Apply perturbations to test set
        test_set = Perturbation(dataset=test_set, config=config)
        test_drop_last = len(test_set) % config.model_config.batch_size_val == 1

        # Check for distributed setup and return DataLoader
        if tools.is_dist_avail_and_initialized():
            test_sampler = DistributedSampler(test_set, shuffle=False)
            test_loader = DataLoader(
                test_set,
                batch_size=config.model_config.batch_size_val,
                num_workers=config.num_workers,
                drop_last=test_drop_last,
                sampler=test_sampler
            )
            return test_loader, test_sampler
        else:
            test_loader = DataLoader(
                test_set,
                batch_size=config.model_config.batch_size_val,
                num_workers=config.num_workers,
                shuffle=False,
                drop_last=test_drop_last
            )
            return test_loader, None

    # Handling train mode
    if config.mode == 'train':
        # Initialize train dataset
        train_set = KittiDataset(config=config, split='train')
        train_set = Perturbation(dataset=train_set, config=config)
        train_drop_last = len(train_set) % config.model_config.batch_size == 1

        # Check for distributed setup and return DataLoader
        if tools.is_dist_avail_and_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.model_config.batch_size,
                num_workers=config.num_workers,
                drop_last=train_drop_last,
                sampler=train_sampler
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.model_config.batch_size,
                num_workers=config.num_workers,
                shuffle=True,
                drop_last=train_drop_last
            )

    # Handling validation dataset
    val_set = KittiDataset(config=config, split='val')
    val_length = len(val_set)

    # Check if perturbation file exists for validation
    if not os.path.exists(config.dataset_config.val_perturb_file):
        if recorder is not None:
            recorder.log_message(">> Validation perturb file doesn't exist, creating one.", level='info')
        __create_perturb_file(config=config, length=val_length, path_to_file=config.dataset_config.val_perturb_file)
    else:
        val_seq = np.loadtxt(config.dataset_config.val_perturb_file, delimiter=',')
        if val_length != val_seq.shape[0]:
            if recorder is not None:
                recorder.log_message(f'>> Incompatible validation length {val_length}!={val_seq.shape[0]}', level='warning')
            __create_perturb_file(config=config, length=val_length, path_to_file=config.dataset_config.val_perturb_file)
            if recorder is not None:
                recorder.log_message('>> Validation perturb file rewritten.', level='info')

    # Apply perturbations to validation set
    val_set = Perturbation(dataset=val_set, config=config)
    val_drop_last = len(val_set) % config.model_config.batch_size_val == 1

    # Check for distributed setup and return DataLoader
    if tools.is_dist_avail_and_initialized():
        val_sampler = DistributedSampler(val_set, shuffle=False)
        val_loader = DataLoader(
            val_set,
            batch_size=config.model_config.batch_size_val,
            num_workers=config.num_workers,
            drop_last=val_drop_last,
            sampler=val_sampler
        )
    else:
        val_loader = DataLoader(
            val_set,
            batch_size=config.model_config.batch_size_val,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=val_drop_last
        )

    # Log successful data loading
    if recorder is not None:
        recorder.log_message(">> Data loaded successfully.", level='info')

    # Return appropriate DataLoaders based on mode and distributed setup
    if config.mode == "train":
        if tools.is_dist_avail_and_initialized():
            return train_loader, val_loader, train_sampler, val_sampler
        else:
            return train_loader, val_loader, None, None
    else:
        if tools.is_dist_avail_and_initialized():
            return None, val_loader, None, val_sampler
        else:
            return None, val_loader, None, None



def __kitti_raw(config: Config, recorder: Optional[Recorder]) -> Union[
        Tuple[DataLoader, Optional[DistributedSampler]],
        Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]],
        Tuple[None, DataLoader, None, Optional[DistributedSampler]]
    ]:
    """
    Loads the raw KITTI dataset and prepares DataLoaders for training, validation, or testing.

    Args:
        config (Config): Configuration object containing dataset and model settings.
        recorder (Optional[Recorder]): Recorder object for logging messages, can be None.

    Returns:
        Union[Tuple[DataLoader, Optional[DistributedSampler]], 
              Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]], 
              Tuple[None, DataLoader, None, Optional[DistributedSampler]]]:
            Depending on the mode and distributed setup, returns the appropriate DataLoaders and Samplers.
    """
    # Handling test mode
    if config.mode == 'test':
        # Initialize test dataset (raw version)
        test_set = KittiDatasetRaw(config=config, split='test')
        test_length = len(test_set)

        # Check if perturbation file exists for test
        if not os.path.exists(config.dataset_config.test_perturb_file):
            if recorder is not None:
                recorder.log_message(">> Validation perturb file doesn't exist, creating one.", level='info')
            __create_perturb_file(config=config, length=test_length, path_to_file=config.dataset_config.test_perturb_file)
        else:
            test_seq = np.loadtxt(config.dataset_config.test_perturb_file, delimiter=',')
            if test_length != test_seq.shape[0]:
                if recorder is not None:
                    recorder.log_message(f'>> Incompatible validation length {test_length}!={test_seq.shape[0]}', level='warning')
                __create_perturb_file(config=config, length=test_length, path_to_file=config.dataset_config.test_perturb_file)
                if recorder is not None:
                    recorder.log_message('>> Validation perturb file rewritten.', level='info')

        # Apply perturbations to test set
        test_set = Perturbation(dataset=test_set, config=config)
        test_drop_last = len(test_set) % config.model_config.batch_size_val == 1

        # Check for distributed setup and return DataLoader
        if tools.is_dist_avail_and_initialized():
            test_sampler = DistributedSampler(test_set, shuffle=False)
            test_loader = DataLoader(
                test_set,
                batch_size=config.model_config.batch_size_val,
                num_workers=config.num_workers,
                drop_last=test_drop_last,
                sampler=test_sampler
            )
            return test_loader, test_sampler
        else:
            test_loader = DataLoader(
                test_set,
                batch_size=config.model_config.batch_size_val,
                num_workers=config.num_workers,
                shuffle=False,
                drop_last=test_drop_last
            )
            return test_loader, None

    # Handling train mode
    if config.mode == 'train':
        # Initialize train dataset (raw version)
        train_set = KittiDatasetRaw(config=config, split='train')
        train_set = Perturbation(dataset=train_set, config=config)
        train_drop_last = len(train_set) % config.model_config.batch_size == 1

        # Check for distributed setup and return DataLoader
        if tools.is_dist_avail_and_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.model_config.batch_size,
                num_workers=config.num_workers,
                drop_last=train_drop_last,
                sampler=train_sampler
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.model_config.batch_size,
                num_workers=config.num_workers,
                shuffle=True,
                drop_last=train_drop_last
            )

    # Handling validation dataset
    val_set = KittiDatasetRaw(config=config, split='val')
    val_length = len(val_set)

    # Check if perturbation file exists for validation
    if not os.path.exists(config.dataset_config.val_perturb_file):
        if recorder is not None:
            recorder.log_message(">> Validation perturb file doesn't exist, creating one.", level='info')
        __create_perturb_file(config=config, length=val_length, path_to_file=config.dataset_config.val_perturb_file)
    else:
        val_seq = np.loadtxt(config.dataset_config.val_perturb_file, delimiter=',')
        if val_length != val_seq.shape[0]:
            if recorder is not None:
                recorder.log_message(f'>> Incompatible validation length {val_length}!={val_seq.shape[0]}', level='warning')
            __create_perturb_file(config=config, length=val_length, path_to_file=config.dataset_config.val_perturb_file)
            if recorder is not None:
                recorder.log_message('>> Validation perturb file rewritten.', level='info')

    # Apply perturbations to validation set
    val_set = Perturbation(dataset=val_set, config=config)
    val_drop_last = len(val_set) % config.model_config.batch_size_val == 1

    # Check for distributed setup and return DataLoader
    if tools.is_dist_avail_and_initialized():
        val_sampler = DistributedSampler(val_set, shuffle=False)
        val_loader = DataLoader(
            val_set,
            batch_size=config.model_config.batch_size_val,
            num_workers=config.num_workers,
            drop_last=val_drop_last,
            sampler=val_sampler
        )
    else:
        val_loader = DataLoader(
            val_set,
            batch_size=config.model_config.batch_size_val,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=val_drop_last
        )

    # Log successful data loading
    if recorder is not None:
        recorder.log_message(">> Data loaded successfully.", level='info')

    # Return appropriate DataLoaders based on mode and distributed setup
    if config.mode == "train":
        if tools.is_dist_avail_and_initialized():
            return train_loader, val_loader, train_sampler, val_sampler
        else:
            return train_loader, val_loader, None, None
    else:
        if tools.is_dist_avail_and_initialized():
            return None, val_loader, None, val_sampler
        else:
            return None, val_loader, None, None



def __create_perturb_file(config: Config, length: int, path_to_file: str) -> None:
    """
    Creates a perturbation file for the NuScenes dataset, containing random transformations
    (both rotation and translation) that can be used for dataset augmentation or evaluation.

    Args:
        config (Config): Configuration object containing dataset settings like max degrees,
                         max translation, and distribution for the perturbation generation.
        length (int): Number of entries (perturbations) to generate in the perturbation file.
        path_to_file (str): Path where the perturbation file will be saved.
    
    Returns:
        None: This function does not return any value. It writes the perturbation data to the specified file.
    """
    # Initialize the perturbation transform object with the given configuration
    transform = UniformTransformSE3(
        max_deg=config.dataset_config.max_deg,  # Maximum rotation degree
        max_tran=config.dataset_config.max_tran,  # Maximum translation magnitude
        distribution=config.dataset_config.distribution,  # Distribution type for perturbations
        mag_randomly=config.dataset_config.mag_randomly  # Whether the magnitude is random
    )

    # Initialize an array to store perturbation data
    perturb_arr = np.zeros([length, 6])  # Array with shape (length, 6) for rotation and translation

    # Generate the perturbations and fill the array
    for i in range(length):
        # Generate a random transformation (rotation + translation)
        perturb_arr[i, :] = transform.generate_transform().cpu().numpy()

    # Save the perturbation array to a CSV file
    np.savetxt(path_to_file, perturb_arr, delimiter=',')
    
    # Add a slight delay to avoid any potential race conditions in file saving
    time.sleep(2)