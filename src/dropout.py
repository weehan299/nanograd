import numpy as np
from src.module import Module
from src.tensor import Tensor

class Dropout(Module):
    
    def __init__(self, p=0.5):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        self.p = p
        self.training = True  # Training mode flag
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            # During inference or when p=0, return input unchanged
            return x
        
        if self.p == 1:
            # When p=1, return zeros
            out_data = np.zeros_like(x.data)
            out = Tensor(out_data, requires_grad=x.requires_grad)
            
            def _backward():
                if x.requires_grad:
                    # Gradient is zero since output is always zero
                    x.grad += np.zeros_like(x.data)
            
            out._backward = _backward
            out._prev = {x}
            return out
        
        # Generate random mask
        # We use > instead of < to match PyTorch's behavior
        mask = np.random.rand(*x.data.shape) > self.p
        
        # Scale by 1/(1-p) to maintain expected value during training
        # This is "inverted dropout" - the standard approach
        scale = 1.0 / (1.0 - self.p)
        
        # Apply mask and scale
        out_data = x.data * mask * scale
        out = Tensor(out_data, requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                # Gradient flows through only the non-dropped elements
                # and is scaled by the same factor
                x.grad += out.grad * mask * scale
        
        out._backward = _backward
        out._prev = {x}
        return out