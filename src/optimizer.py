
import numpy as np
from typing import List, Optional
from src.tensor import Tensor
from src.module import Module

class Adam:
    """Adam optimizer implementation following PyTorch style."""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """
        Initialize Adam optimizer.
        
        Args:
            parameters: Iterable of parameters to optimize (typically model.parameters())
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps: Term added to denominator for numerical stability (default: 1e-8)
            weight_decay: Weight decay (L2 penalty) (default: 0.0)
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Convert parameters to list if it's a generator
        self.param_groups = [{'params': list(parameters)}]
        
        # Initialize state for each parameter
        self.state = {}
        self.t = 0  # time step
        
        for param in self.param_groups[0]['params']:
            self.state[id(param)] = {
                'm': np.zeros_like(param.data),  # First moment estimate
                'v': np.zeros_like(param.data),  # Second moment estimate
            }
    
    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for param in self.param_groups[0]['params']:
            if param.requires_grad:
                param.grad = np.zeros_like(param.data)
    
    def step(self):
        """Perform a single optimization step."""
        self.t += 1
        
        for param in self.param_groups[0]['params']:
            if not param.requires_grad or param.grad is None:
                continue
                
            grad = param.grad
            
            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            param_state = self.state[id(param)]
            m, v = param_state['m'], param_state['v']
            
            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            
            # Store updated moments
            param_state['m'] = m
            param_state['v'] = v
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)