import os
import torch
from types import SimpleNamespace
import json

def load_checkpoint(model, epoch=None):
    if epoch is None:
        return None
    elif epoch == "best":
        checkpoint_path = f"checkpoints/{model.config.name}/best.pt"
    else:
        checkpoint_path = f"checkpoints/{model.config.name}/epoch_{epoch}.pt"
    if not os.path.exists(checkpoint_path):
        return None
    return torch.load(checkpoint_path, weights_only=False, map_location="cpu")

def load_config(config_name):
    
    config_dir = f"configs/{config_name}.json"
    default_config_dir = f"configs/default.json"
    
    if not os.path.exists(config_dir):
        raise ValueError(f"Config file not found: {config_dir}")
    
    def dict_to_namespace(d, default=None):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        if default:
            for k, v in default.items():
                if k not in d:
                    d[k] = v
        return SimpleNamespace(**d)
    
    default_config = json.load(open(default_config_dir))
    config = json.load(open(config_dir))
    config = dict_to_namespace(config, default_config)
    config.name = config_name
    
    return config