import os
import torch
from types import SimpleNamespace
import json

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "../checkpoints")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../configs")

def load_most_recent_checkpoint(model):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model.config.name}.pt")
    if not os.path.exists(checkpoint_path):
        return None
    return torch.load(checkpoint_path, weights_only=False, map_location="cpu")

def load_config(config_name):
    
    config_dir = os.path.join(CONFIG_DIR, f"{config_name}.json")
    
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
    
    default_config = json.load(os.path.join(CONFIG_DIR, "default.json"))
    config = dict_to_namespace(json.load(open(config_dir)), default_config)
    config.name = config_name
    
    return config