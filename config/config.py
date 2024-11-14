import os
import json
from typing import Optional, Dict
import yaml
from pathlib import Path

class Config(object):
    """
    Config class that manages all configuration options for training, evaluation, and logging.

    Attributes:
        root (str): Root path of the project.
        count (int): Run count for logging or experiments.
        project_name (str): Name of the project.
        group_name (str): Group name for organizing experiments.
        use_sweeps (bool): Whether to use sweep configurations.
        sweep_config (Optional[dict]): Sweep configuration if enabled.
        seed (int): Random seed for reproducibility.
        gpu (int): GPU index to use.
        distributed (bool): Flag for distributed training.
        model_config (ModelConfig): Configuration for the model.
        dataset_config (DataConfig): Configuration for the dataset.
        save_path (str): Path to save results, logs, and checkpoints.
    """
    def __init__(self, args):
        # Initialize general settings
        self.root = args.root
        self.count = args.count
        self.project_name = args.project_name
        self.group_name = args.group_name
        self.use_sweeps = args.use_sweeps
        
        # Viz
        self.visualize = args.visualize

        # Load sweep configuration if enabled
        self.sweep_config = None
        if args.use_sweeps:
            self.sweep_config = yaml.safe_load(Path("/workspace/config/sweep_config.yaml").read_text())
        
        # General training options
        self.seed = 1
        self.gpu = 0
        self.rank = 0
        self.world_size = 1
        self.distributed = False
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        self.num_workers = args.num_workers
        
        # Load model configuration
        model_config = self.__read_json_file(os.path.join(args.root, 'config', 'model_config.json'))
        if args.model == 'calibvit':
            self.model_config = ModelConfig(model_config[args.version][args.dataset])
        else:
            self.model_config = ModelConfig(model_config[args.dataset])
        
        # Update paths for model checkpoints and pretrained weights
        self.model_config.path_to_checkpoint = os.path.join(self.root, self.model_config.path_to_checkpoint)
        self.model_config.path_to_pretrained_backbone = os.path.join(self.root, self.model_config.path_to_pretrained_backbone)
        self.model_config.path_to_pretrained_ca = os.path.join(self.root, self.model_config.path_to_pretrained_ca)
        
        # Training configurations
        self.mode = args.mode
        self.val_frequency = 100
        
        # Override model settings with command-line arguments if provided
        if args.epochs is not None:
            self.model_config.epochs = args.epochs
        if args.batch_size is not None:
            self.model_config.batch_size = args.batch_size
        if args.lr is not None:
            self.model_config.lr = args.lr
        if args.inner_iter is not None:
            self.model_config.inner_iter = args.inner_iter
        
        # Logging configurations
        self.save_path = os.path.join(args.root, 'results', args.model)
        if args.model == 'calibvit':
            self.save_path = os.path.join(self.save_path, args.version)
        self.checkpoint_path = os.path.join(self.save_path, 'checkpoint')
        self.prediction_path = os.path.join(self.save_path, 'preds')
        self.log_frequency = 100
        self.train_result_frequency = 10
        
        # Load dataset configuration
        self.dataset = args.dataset
        dataset_config = self.__read_json_file(os.path.join(args.root, 'config', 'dataset_config.json'))
        self.dataset_config = DataConfig(dataset_config[args.dataset])
        self.dataset_path = os.path.join(args.root, self.dataset_config.dataset_path)
        
        # Update dataset configuration with command-line arguments
        if args.max_deg is not None:
            self.dataset_config.max_deg = args.max_deg
        if args.max_tran is not None:
            self.dataset_config.max_tran = args.max_tran
        self.dataset_config.distribution = args.distribution
        
        # Generate paths for validation and test perturbation files
        self.dataset_config.val_perturb_file = os.path.join(self.root, 'weights', 'validation_decalib', f"{self.dataset}_val_{self.dataset_config.distribution}_{self.dataset_config.max_deg}_{self.dataset_config.max_tran}.csv")
        self.dataset_config.test_perturb_file = os.path.join(self.root, 'weights', 'test_decalib', f"{self.dataset}_test_{self.dataset_config.distribution}_{self.dataset_config.max_deg}_{self.dataset_config.max_tran}.csv")
        self.dataset_config.image_size = self.model_config.image_size
    
    def update_config(self, wandb_config: dict):
        """
        Updates the model and dataset configurations with parameters from a wandb sweep configuration.

        Args:
            wandb_config (dict): Dictionary containing updated hyperparameters.
        """
        model_keys = ['lr', 'epochs', 'momentum', 'decay', 'ca_layer_dropout', 'ca_layer_drprate', 'fc_layer_dropout', 'optimizer']
        dataset_keys = ['pcd_min_samples', 'pooling_size']
        
        for key in model_keys:
            if key in wandb_config:
                setattr(self.model_config, key, wandb_config[key])
        
        for key in dataset_keys:
            if key in wandb_config:
                setattr(self.dataset_config, key, wandb_config[key])
    
    @staticmethod
    def __read_json_file(file_path: str) -> dict:
        """
        Reads a JSON file and returns its content.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: Parsed JSON content.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r') as file:
            return json.load(file)


class ModelConfig:
    """Class for storing model-specific configurations."""
    def __init__(self, dictionary: Dict):
        for k, v in dictionary.items():
            setattr(self, k, v)


class DataConfig:
    """Class for storing dataset-specific configurations."""
    def __init__(self, dictionary: Dict):
        for k, v in dictionary.items():
            setattr(self, k, v)


def config_to_dict(obj) -> dict:
    """
    Converts an object's attributes into a dictionary.

    Args:
        obj: The object to convert.

    Returns:
        dict: Dictionary representation of the object.
    """
    if isinstance(obj, list):
        return [config_to_dict(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return {k: config_to_dict(v) for k, v in obj.__dict__.items()}
    return obj


def dict_to_config(d: dict, cls):
    """
    Converts a dictionary into an object of the specified class.

    Args:
        d (dict): The dictionary to convert.
        cls: The target class.

    Returns:
        An instance of the specified class with attributes set from the dictionary.
    """
    obj = cls.__new__(cls)  # Create an instance without invoking __init__
    for key, value in d.items():
        setattr(obj, key, value)
    return obj
