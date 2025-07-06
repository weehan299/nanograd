import numpy as np
from module import Module
from tensor import Tensor

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
    


# Example usage and test
if __name__ == "__main__":
    # Test LayerNorm with different configurations
    print("Testing LayerNorm implementation:")
    
    # Test 1: Simple 1D case
    print("\nTest 1: 1D LayerNorm")
    np.random.seed(42) 
    #x = Tensor(np.random.randn(3, 4), requires_grad=True)
    #ln = LayerNorm(4)
    x = Tensor(np.array([[1,2,3,4],[4,5,6,7],[2,2,2,2]]), requires_grad=True)
    ln = LayerNorm(4)
    
    print(f"Input shape: {x.data.shape}")
    print(f"Input:\n{x.data}")
    
    y = ln(x)
    print(f"Output shape: {y.data.shape}")
    print(f"Output:\n{y.data}")
    
    # Check normalization (mean should be ~0, std should be ~1)
    print(f"Output mean per sample: {np.mean(y.data, axis=1)}")
    print(f"Output std per sample: {np.std(y.data, axis=1)}")
    
    # Test backward pass
    loss = y.sum()
    loss.backward()
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Weight gradient shape: {ln.weight.grad.shape}")
    print(f"Bias gradient shape: {ln.bias.grad.shape}")
    
    # Test 2: 2D case (like for images)
    print("\nTest 2: 2D LayerNorm")
    x2 = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    ln2 = LayerNorm((3, 4))
    
    print(f"Input shape: {x2.data.shape}")
    print(f"Input:\n{x2.data}"  )
    y2 = ln2(x2)
    print(f"Output shape: {y2.data.shape}")
    print(f"Output:\n{y2.data}")
    
    # Test 3: Without learnable parameters
    print("\nTest 3: LayerNorm without learnable parameters")
    ln3 = LayerNorm(4, elementwise_affine=False)
    y3 = ln3(x)
    print(f"Output mean per sample: {np.mean(y3.data, axis=1)}")
    print(f"Output std per sample: {np.std(y3.data, axis=1)}")
    print(f"Has learnable parameters: {ln3.elementwise_affine}")