import random
import numpy as np
import torch
import json
from pathlib import Path
import inspect

def set_seed(seed):
    """
    Set random seed for reproducibility across libraries.
    
    Args:
    - seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_config_to_json(config_module, output_file_path):
    """
    Save all attributes of a config module to a JSON file.
    
    Parameters:
        config_module (module): The module containing configuration variables.
        output_file (str): The path to the output JSON file.
    """
    # Helper function to serialize non-serializable objects to strings
    def serialize_value(value):
        if isinstance(value, (Path, torch.device, type)):
            return str(value)  # Convert Path and device objects to strings
        return value  # Leave other types unchanged

    # Create a dictionary from the module's attributes
    config_dict = {}
    for name, value in inspect.getmembers(config_module):
        if not name.startswith("__") and not inspect.ismodule(value) and not inspect.isfunction(value):
            config_dict[name] = serialize_value(value)

    # Save to a JSON file
    with open(output_file_path, "w") as f:
        json.dump(config_dict, f, indent=4)

def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from a checkpoint file.
    
    Args:
    - model (torch.nn.Module): The model to load weights into.
    - checkpoint_path (str or Path): The path to the checkpoint file.
    - device (str or torch.device): The device to load the model on.
    
    Returns:
    - model (torch.nn.Module): The model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model

def count_model_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
    - model (torch.nn.Module): The model to count parameters for.
    
    Returns:
    - int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
