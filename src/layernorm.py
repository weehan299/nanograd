
import numpy as np
from src.module import Module
from src.tensor import Tensor

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Tensor(np.ones(normalized_shape), requires_grad=True)
            self.bias = Tensor(np.zeros(normalized_shape), requires_grad=True)
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:

        # Calculate dimensions to normalize over (last len(normalized_shape) dimensions)
        norm_dims = tuple(range(x.data.ndim - len(self.normalized_shape), x.data.ndim))
        mean_data = np.mean(x.data, axis=norm_dims, keepdims=True)
        var_data = np.var(x.data, axis=norm_dims, keepdims=True)
        mean = Tensor(mean_data, requires_grad=False)
        var = Tensor(var_data, requires_grad=False)
        x_centered = x - mean
        
        std = var + self.eps
        std_sqrt_data = np.sqrt(std.data)
        std_sqrt = Tensor(std_sqrt_data, requires_grad=False)
        
        # Division operation: x_centered / std_sqrt
        normalized = x_centered / std_sqrt
        
        # Apply learnable parameters if enabled
        if self.elementwise_affine:
            normalized = normalized * self.weight + self.bias
            
        return normalized
    

