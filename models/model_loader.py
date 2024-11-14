import os 
import torch
from torch import nn
from typing import Union
from config import ModelConfig
from logger import Recorder
from .calibnet import CalibNet
from .calibvit import CalibViTV1, CalibViTV2


def load_model(config:ModelConfig, recorder: Union[Recorder, None]) -> nn.Module:

    model_name = config.model

    if model_name == "calibnet":
        model= CalibNet()

    elif model_name == "calibvit":
        if config.version == "v1":
            model = CalibViTV1(config=config)
        
        elif config.version == "v2":
            model = CalibViTV2(config=config)
        else:
            if recorder is not None:
                recorder.log_message("Invalid version of CalibVit", level="error")
            
    else:
        if recorder is not None:
            recorder.log_message(">> Model not registered. Refere to documentation to add the new model", level='error')
        
       
    if recorder is not None:
        recorder.log_message(">> Successfully loaded {}".format(model_name), level="info")
        
    return model 


def load_checkpoint(config:ModelConfig, model:torch.nn.Module) -> torch.nn.Module:
    
    if not os.path.isfile(config.path_to_checkpoint):
        raise FileNotFoundError('>> Checkpoint file not found: {}'.format(config.path_to_checkpoint))

    checkpoint_data = torch.load(config.path_to_checkpoint, map_location='cpu')

    checkpoint_data_model = checkpoint_data['model']
    msg = model.load_state_dict(checkpoint_data_model, strict=True) #(not self.config.mode=="train"
    print(">>> " + str(msg))

    return model