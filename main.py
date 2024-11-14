import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import torch
from torch import nn
import argparse
import os
import datetime
import time
import numpy as np
from config import Config
import models
import logger
import tools
from training import Trainer, Test
import pprint as pp
import wandb
from typing import Optional
import yaml
from pathlib import Path


class Experiment:
    """
    Manages the workflow of an experiment, including initialization, training, validation, and testing.
    """

    def __init__(self, config: Config, wandb_recorder: Optional[logger.WandbRecorder]):
        """
        Initializes the Experiment class.

        Args:
            config (Config): Configuration object containing experimental settings.
            wandb_recorder (Optional[logger.WandbRecorder]): Optional Wandb logger for experiment tracking.
        """
        self.config = config  # Configuration object for the experiment
        self.count = config.count  # Keeps track of experiment count (useful for logging or checkpointing)
        self.epoch_start = 0  # Initialize epoch counter to 0

        # Initialize the logger for recording messages and results
        self.recorder = None
        if tools.is_main_process():  # Only initialize the logger on the main process
            self.recorder = logger.Recorder(self.config)
            # Log the full configuration details
            self.recorder.log_message(
                '>> Config \n %s' % pp.pformat(self.__get_object_vars(self.config)), 
                level='info'
            )
            self.count += 1  # Increment count for the new experiment

        # Initialize the Wandb recorder for experiment tracking
        self.wandb_recorder = wandb_recorder

        # Initialize the model
        self.model = self._initModel()

        # Initialize trainer or tester based on the mode
        if self.config.mode in ['train', 'val']:
            # Training/validation mode
            self.trainer = Trainer(
                config=self.config, 
                model=self.model, 
                recorder=self.recorder, 
                wand_recorder=self.wandb_recorder
            )
            # Log model structure and criteria to Wandb (only on main process)
            if self.wandb_recorder is not None and tools.is_main_process():
                self.wandb_recorder.wandb_logger.watch(
                    self.model, 
                    self.trainer.criterion, 
                    log="all", 
                    log_graph=True
                )
        else:
            # Testing mode
            self.tester = Test(
                config=self.config, 
                model=self.model, 
                recorder=self.recorder
            )

        # Load a checkpoint if available
        self._loadCheckpoint()

        # Execute the experiment
        self.run()

    def __get_object_vars(self, obj):
        """
        Recursively retrieves attributes of an object and its nested objects as a dictionary.

        Args:
            obj: Object whose attributes are to be retrieved.

        Returns:
            dict: A dictionary containing all attributes of the object.
        """
        if not hasattr(obj, '__dict__'):
            return obj  # Base case: Return the object itself if it has no attributes.

        # Recursively retrieve attributes
        attributes = {
            key: self.__get_object_vars(value) if hasattr(value, '__dict__') else value
            for key, value in obj.__dict__.items()
        }

        # Handle special cases or sensitive attributes (e.g., sweep configurations)
        attributes['sweep_config'] = None

        return attributes

    def _initModel(self) -> nn.Module:
        """
        Initializes the model based on the provided configuration.

        Returns:
            nn.Module: The initialized model object.
        """
        # Use the configuration to load the model
        model = models.load_model(self.config.model_config, self.recorder)

        # Log model details if the recorder is initialized
        if self.recorder is not None:
            self.recorder.log_message(f'Model initialized: {model}', level='info')

        return model

    
    def _loadCheckpoint(self):
        """
        Loads model weights and optimizer state from a checkpoint or pretrained weights.

        Depending on the configuration, this function:
        - Loads pretrained ResNet backbones for feature extraction.
        - Loads pretrained Vision Transformer (ViT) weights for cross-attention blocks.
        - Resumes training or testing from a specified checkpoint.

        Raises:
            FileNotFoundError: If the specified checkpoint file is not found.
        """
        # Load pretrained backbones if specified
        if self.config.model_config.pretrained_backbone:
            if self.config.model_config.model == 'calibvit':

                # Load pretrained ResNet backbones
                if self.config.model_config.version == 'v1':
                    checkpoint = torch.load(self.config.model_config.path_to_pretrained_backbone)
                    pretrained_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

                    # Separate weights for RGB and depth networks
                    pretrained_dict_rgb, pretrained_dict_depth = {}, {}
                    for key in pretrained_dict.keys():
                        if "rgb" in key:
                            pretrained_dict_rgb[key.replace('rgb_resnet.', '')] = pretrained_dict[key]
                        if "rgb_resnet" in key:
                            pretrained_dict_depth[key.replace('rgb_resnet.', '1.')] = pretrained_dict[key]

                    # Load weights into the model
                    self.model.feature_extractor.rgb_resnet.load_state_dict(pretrained_dict_rgb)

                    # Allow gradients for all backbone parameters
                    for param in self.model.feature_extractor.parameters():
                        param.requires_grad = True

                    if self.recorder:
                        self.recorder.log_message('>> Loaded pretrained ResNet backbones successfully')

                # Load pretrained cross-attention (CA) weights if specified
                if self.config.model_config.pretrained_ca:
                    checkpoint = torch.load(self.config.model_config.path_to_pretrained_ca)
                    pretrained_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                    adjusted_pretrained_dict = {}

                    # Adjust pretrained weights for Q, K, V split in cross-attention blocks
                    for k, v in pretrained_dict.items():
                        if 'qkv' in k:
                            qkv_size = v.size(0)
                            size_per_head = qkv_size // 3
                            q = v[:size_per_head]  # Query weights
                            kv = torch.cat((v[size_per_head:2 * size_per_head], v[2 * size_per_head:]), dim=0)  # Key and Value weights
                            new_key_q = k.replace('qkv', 'q').replace('blocks.', '')
                            new_key_kv = k.replace('qkv', 'kv').replace('blocks.', '')
                            adjusted_pretrained_dict[new_key_q] = q
                            adjusted_pretrained_dict[new_key_kv] = kv
                        else:
                            new_key = k.replace('blocks.', '')
                            adjusted_pretrained_dict[new_key] = v

                    # Load adjusted weights into ViT cross-attention blocks
                    self.model.vit_rgb.blocks.load_state_dict(adjusted_pretrained_dict, strict=False)

                    # Allow gradients for all cross-attention block parameters
                    for param in self.model.vit_rgb.blocks.parameters():
                        param.requires_grad = True

                    if self.recorder:
                        self.recorder.log_message('>> Loaded pretrained ViT for cross-attention blocks successfully')

        # Load checkpoint if pretrained backbone is not specified
        if not self.config.model_config.pretrained_backbone and self.config.model_config.path_to_checkpoint:
            if self.recorder:
                mode_msg = 'Resume training' if self.config.mode == 'train' else 'Checkpoint loaded'
                self.recorder.log_message(f'>> {mode_msg} from {self.config.model_config.path_to_checkpoint}', level='info')

            if not os.path.isfile(self.config.model_config.path_to_checkpoint):
                raise FileNotFoundError(f'>> Checkpoint file not found: {self.config.model_config.path_to_checkpoint}')

            # Load checkpoint data
            checkpoint_data = torch.load(self.config.model_config.path_to_checkpoint, map_location='cpu')
            checkpoint_data_model = checkpoint_data['model']
            msg = self.model.load_state_dict(checkpoint_data_model, strict=False)

            if self.recorder:
                self.recorder.log_message(f'Checkpoint load message: {msg}', level='info')

            # Resume optimizer state and epoch count if in training mode
            if self.config.mode == "train":
                self.trainer.optimizer.load_state_dict(checkpoint_data["optimizer"])
            self.epoch_start = checkpoint_data['epoch'] + 1


    def run(self):
        """
        Execute the experiment based on the mode (train, val, or test).
        
        - Training: Runs training for the configured number of epochs, with validation and checkpoint saving.
        - Validation: Evaluates the model and logs the results.
        - Testing: Runs inference on the test set and saves results.
        """
        t_start = time.time()

        # Validation Mode
        if self.config.mode == 'val':
            self.trainer.run(self.epoch_start, mode='val')

            cost_time = time.time() - t_start
            if self.recorder:
                self.recorder.logger.info(f'==== Total cost time: {datetime.timedelta(seconds=cost_time)}')
            return

        # Test Mode
        if self.config.mode == 'test':
            self.tester.run(
                self.epoch_start,
                print_results=True,
                save_results_path=self.config.prediction_path
            )

            cost_time = time.time() - t_start
            if self.recorder:
                self.recorder.logger.info(f'==== Total cost time: {datetime.timedelta(seconds=cost_time)}')
            return

        # Training Mode
        best_val_result = None
        for epoch in range(self.epoch_start, self.config.model_config.epochs):
            # Train for one epoch
            train_result = self.trainer.run(epoch, mode='train')

            # Perform validation after each epoch
            val_result = self.trainer.run(epoch, mode='val')

            # Check for improvements in validation results
            if self.recorder:
                self.recorder.logger.info(f'---- Best result after Epoch {epoch + 1} ----')
            if best_val_result is None:
                best_val_result = val_result

            for metric, value in val_result.items():
                # Update the best result if the current value is better
                if value <= best_val_result[metric]:
                    if self.recorder:
                        self.recorder.logger.info(f'Got better {metric} model with error: {value}')

                    # Save the best model checkpoint
                    checkpoint_path = os.path.join(
                        self.config.checkpoint_path,
                        f'best_{metric}_{self.config.dataset}_{self.config.model_config.model}_{self.count}.pth'
                    )

                    best_val_result[metric] = value

                    checkpoint_data = {
                        'model': self.model.state_dict(),
                        'optimizer': self.trainer.optimizer.state_dict(),
                        'epoch': epoch,
                        'model_config': self.config.model_config.__dict__,
                        metric: value,
                    }

                    if self.trainer.fp16_scaler:
                        checkpoint_data['fp16_scaler'] = self.trainer.fp16_scaler.state_dict()

                    torch.save(checkpoint_data, checkpoint_path)

            # Log the best results so far
            if best_val_result:
                log_str = '>>> Best Result: '
                log_str += ' '.join([f'{k}: [{v}]' for k, v in best_val_result.items()])
                if self.recorder:
                    self.recorder.log_message(log_str, level='info')

        # Log total execution time
        cost_time = time.time() - t_start
        if self.recorder:
            self.recorder.log_message(f'=== Total cost time: {datetime.timedelta(seconds=cost_time)}', level='info')

        # Finish WandB logging
        if self.wandb_recorder:
            self.wandb_recorder.finish()

    
def run_experiment(config=None):
    """
    Initialize and execute an experiment based on the configuration.
    Handles distributed processes, configuration updates, and WandB logging.

    Args:
        config: Sweep configuration for hyperparameter optimization.
    """
    # WandB Recorder and Config Handling for Main Process
    if tools.is_main_process():
        # Initialize WandB Recorder with the configuration
        wandb_recorder = logger.WandbRecorder(config=global_config, sweep_config=config)

        # Save the sweep configuration to a temporary YAML file
        sweep_config = wandb_recorder.wandb_logger.config
        temp_config_path = "/workspace/temp_config.yaml"
        with open(temp_config_path, "w") as yaml_file:
            yaml.dump(vars(sweep_config)['_items'], yaml_file)

    else:  # Handle Non-Main Processes
        # Load the sweep configuration from the temporary file
        temp_config_path = "/workspace/temp_config.yaml"
        sweep_config = yaml.safe_load(Path(temp_config_path).read_text())
        global_config.update_config(sweep_config)
        wandb_recorder = None  # WandB logging is only active in the main process

    # Update Global Configuration
    global_config.update_config(sweep_config)

    # Run the Experiment
    Experiment(config=global_config, wandb_recorder=wandb_recorder)


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Run camera-LiDAR calibration experiments.")

    # General configuration
    parser.add_argument('--root', default="/workspace", type=str, help='Root folder path')
    parser.add_argument('--project_name', default="camera-lidar", type=str, help='WandB project name')
    parser.add_argument('--group_name', default="test_experiment", type=str, help='Experiment group name')
    parser.add_argument('--use_sweeps', default=False, action='store_true', help='Enable WandB hyperparameter sweeps')
    parser.add_argument('--sweep_id', default=None, type=str, help='WandB sweep ID')
    parser.add_argument('--count', default=0, type=int, help='Number of sweep runs to execute')

    # Model settings
    parser.add_argument('--model', default='calibvit', type=str, choices=['calibvit', 'calibnet'], help='Model type')
    parser.add_argument('--version', default='v1', type=str, choices=['v1', 'v2'], help='Model version')
    parser.add_argument('--inner_iter', default=1, type=int, help='Number of inner iterations')

    # Dataset settings
    parser.add_argument('--dataset', default='nuscenes', type=str, choices=['nuscenes', 'kitti'], help='Dataset name')
    parser.add_argument('--max_deg', default=10, type=int, choices=range(1, 16), help='Max decalibration rotation (degrees)')
    parser.add_argument('--max_tran', default=0.25, type=float, help='Max decalibration translation (meters)')
    parser.add_argument('--distribution', default='uniform', type=str, choices=['uniform', 'gaussian', 'inverse_gaussian'], help='Decalibration distribution')

    # Mode
    parser.add_argument('--mode', default='test', choices=['train', 'val', 'test'], help='Mode of operation')

    # Training settings
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader workers')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')

    # Logging
    parser.add_argument('--use_wandb', default=False, action='store_true', help='Log data to WandB')
    parser.add_argument('--visualize', default=False, action='store_true', help='Visualize predictions')

    # Parse arguments
    args = parser.parse_args()

    # Load global configuration
    global global_config
    global_config = Config(args)

    # Initialize distributed training
    rank = tools.init_distributed_mode(global_config)
    torch.distributed.barrier()

    # Set random seed for reproducibility
    torch.manual_seed(global_config.seed)
    torch.cuda.manual_seed(global_config.seed)
    np.random.seed(global_config.seed)
    torch.cuda.set_device(global_config.gpu)
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    # Hyperparameter Sweeps
    if global_config.use_sweeps:
        if tools.is_main_process():
            wandb.agent(
                args.sweep_id, 
                run_experiment, 
                count=args.count, 
                project=global_config.project_name
            )
        else:
            time.sleep(60)  # Allow main process to set up WandB
            run_experiment()
    else:
        # Run the experiment directly
        wandb_recorder = None
        if tools.is_main_process() and global_config.mode == "train" and args.use_wandb:
            wandb_recorder = logger.WandbRecorder(config=global_config, sweep_config=None)
        Experiment(global_config, wandb_recorder)
