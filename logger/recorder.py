import os
import logging
import sys
from typing import Any
from config import Config

class Recorder:
    def __init__(self, config: Config) -> None:
        """
        Initialize a Recorder instance.

        Args:
            config (Any): Configuration settings object.
            save_path (str): Path to save logs and checkpoints.
            use_tensorboard (bool): Flag to use TensorBoard for logging. Default is True.
        """
        print('>> Init a recorder at ', config.save_path)
        self.save_path = config.save_path
        self.config = config
        self.log_path = os.path.join(self.save_path, 'log')
        self.checkpoint_path = os.path.join(self.save_path, 'checkpoint')
        self.prediction_path = os.path.join(self.save_path, 'preds')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.prediction_path, exist_ok=True)

        # Initialize logger
        self.logger = self._init_logger()

        # Save configuration
        if self.config.mode =='train':
            self._save_config()

    def _init_logger(self): # -> logging.Logger:
        """
        Initialize the logger.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger('console')
        logger.propagate = False
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        
        # File log
        file_handler = logging.FileHandler(os.path.join(self.log_path, 'console.log'))
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)

        # Console log
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        return logger

    def _save_config(self) -> None:
        """
        Save configuration settings to a file.
        """
        with open(os.path.join(self.log_path, 'settings.log'), 'w') as f:
            for k, v in self.config.__dict__.items():
                f.write(f'{k}: {v}\n')

    def log_message(self, message: str, level: str = 'info') -> None:
        """
        Log a message at the specified logging level.

        Args:
            message (str): Message to log.
            level (str): Logging level ('info', 'debug', 'warning', 'error', 'critical'). Default is 'info'.
        """
        level = level.lower()
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)
        else:
            self.logger.info(message)

    def log_metrics(self, metrics: dict, epoch: int, mode: str = 'train') -> None:
        """
        Log metrics to the logger.

        Args:
            metrics (dict): Dictionary of metrics to log.
            epoch (int): Current epoch number.
            mode (str): Mode of logging ('train', 'val', 'test'). Default is 'train'.
        """
        log_msg = f">>> Epoch [{epoch}] [{mode}]  " + ', '.join([f'{k}: [{v:.4f}]' for k, v in metrics.items()])
        self.logger.info(log_msg)


