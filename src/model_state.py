import numpy as np
import json
import os
from typing import Dict, Any
from src.tensor import Tensor
from src.gptlanguage import GPTLanguageModel


class ModelState:
    """Utility class for saving and loading model weights"""
    
    @staticmethod
    def save_model(model, filepath: str, additional_info: Dict[str, Any] = None):
        """
        Save model weights and configuration to a file.
        Args:
            model: The model to save
            filepath: Path to save the model to
            additional_info: Additional information to save (like training config)
        """
        state_dict = {}
        
        # Save model hyperparameters
        state_dict['config'] = {
            'vocab_size': model.vocab_size,
            'n_embd': model.n_embd,
            'n_head': model.n_head,
            'n_layer': model.n_layer,
            'block_size': model.block_size,
            'dropout': model.dropout,
            'type': 'GPTLanguageModel'
        }
        
        # Save model weights
        state_dict['weights'] = {}
        for name, param in ModelState._get_named_parameters(model):
            state_dict['weights'][name] = param.data.tolist()
        
        # Save additional info if provided
        if additional_info:
            state_dict['additional_info'] = additional_info
        
        # Create parent directory if it exists in the path
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str):
        """
        Load model weights and configuration from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Tuple of (model, additional_info)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Extract config
        config = state_dict['config']
        
        # Create model with saved config
        model = GPTLanguageModel(
            vocab_size=config['vocab_size'],
            n_embd=config['n_embd'],
            n_head=config['n_head'],
            n_layer=config['n_layer'],
            block_size=config['block_size'],
            dropout=config['dropout']
        )
        
        # Load weights
        weights = state_dict['weights']
        ModelState._load_weights(model, weights)
        
        # Get additional info if available
        additional_info = state_dict.get('additional_info', {})
        
        print(f"Model loaded from {filepath}")
        return model, additional_info
    
    @staticmethod
    def _get_named_parameters(model, prefix=''):
        """Recursively get all named parameters from a model"""
        params = []
        
        # Get parameters from this module
        for name, param in model._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            params.append((full_name, param))
        
        # Get parameters from submodules
        for name, module in model._modules.items():
            new_prefix = f"{prefix}.{name}" if prefix else name
            params.extend(ModelState._get_named_parameters(module, new_prefix))
        
        return params
    
    @staticmethod
    def _load_weights(model, weights_dict, prefix=''):
        """Recursively load weights into a model"""
        # Load parameters for this module
        for name, param in model._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name in weights_dict:
                param.data = np.array(weights_dict[full_name], dtype=np.float32)
        
        # Load parameters for submodules
        for name, module in model._modules.items():
            new_prefix = f"{prefix}.{name}" if prefix else name
            ModelState._load_weights(module, weights_dict, new_prefix)
    
    @staticmethod
    def model_exists(filepath: str) -> bool:
        """Check if a model file exists"""
        return os.path.exists(filepath)


# Utility functions for easy use
def save_model(model, filepath: str, additional_info: Dict[str, Any] = None):
    """Convenience function to save a model"""
    ModelState.save_model(model, filepath, additional_info)

def load_model(filepath: str):
    """Convenience function to load a model"""
    return ModelState.load_model(filepath)

def model_exists(filepath: str) -> bool:
    """Convenience function to check if model exists"""
    return ModelState.model_exists(filepath)
