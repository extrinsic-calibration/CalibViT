import wandb
from typing import Dict, List, Any
from config import Config 

class WandbRecorder:
    def __init__(self, config: Config,  sweep_config) -> None:
        """
        Initialize the WandbRecorder.
        
        Args:
           config : Configuration settings object
        """
        if sweep_config is not None: 
            dict_config =sweep_config
        else:
            dict_config = {**config.__dict__, **config.dataset_config.__dict__,  **config.model_config.__dict__}
            dict_config.pop('dataset_config')
            dict_config.pop('model_config')
        
        self.wandb_logger = wandb.init(project=config.project_name, group=config.group_name,
                                config=dict_config 
                                )
       
        self.train_table = wandb.Table(columns=['Epoch', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'Mean_T', 'Mean_R', 'STD_T', 'STD_R', 'Recall'])
        self.val_table = wandb.Table(columns=['Epoch', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'Mean_T', 'Mean_R', 'STD_T', 'STD_R', 'Recall'])

    def log_losses(self, losses: Dict[str, float], mode: str = 'train', epoch:int=0) -> None:
        """
        Log losses to Wandb.
        
        Args:
            losses (Dict[str, float]): Dictionary of loss names and values.
            mode (str): Mode of logging, 'train' or 'val'. Default is 'train'.
            epoch (int) : Current epoch 
        """
        try:
            temp = {f"{mode}/{k}": v for k, v in losses.items()}
            if mode == 'train':
                temp['epoch'] = epoch
            self.wandb_logger.log(temp)
        except Exception as e:
            print(f"Error logging losses to Wandb: {e}")

    def log_table(self, data: List[Any], mode: str) -> None:
        """
        Log data to the appropriate table.
        
        Args:
            data (List[Any]): List of data to log.
            mode (str): Mode of logging, 'train' or 'val'.
        """
        try:
            if mode == 'train':
                self.train_table.add_data(*data)
                self.wandb_logger.log({'train_table': self.train_table})
            else:
                self.val_table.add_data(*data)
                self.wandb_logger.log({'val_table': self.val_table})
        except Exception as e:
            print(f"Error logging table to Wandb: {e}")

    def finish(self) -> None:
        """Finish the Wandb run."""
        self.wandb_logger.finish()
