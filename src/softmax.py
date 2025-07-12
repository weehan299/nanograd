
import numpy as np
from src.tensor import Tensor

def softmax(x: Tensor, dim=-1) -> Tensor:
    """
    Softmax function with proper gradient flow.
    """
    # Subtract max for numerical stability
    # x_max should NOT require gradients since it's just for numerical stability
    x_max = Tensor(np.max(x.data, axis=dim, keepdims=True), requires_grad=False)
    x_shifted = x - x_max
    
    # Compute exp - this maintains gradient flow from x_shifted
    exp_x = exp(x_shifted)
    
    # Sum along specified dimension - this maintains gradient flow from exp_x
    sum_exp = exp_x.sum(dim=dim, keepdims=True)
    
    # Divide - this maintains gradient flow from both exp_x and sum_exp
    out = exp_x / sum_exp
    return out

def exp(x: Tensor) -> Tensor:
    """Element-wise exponential function with proper gradient flow."""
    out_data = np.exp(x.data)
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    def _backward():
        if x.requires_grad:
            # d/dx exp(x) = exp(x)
            x.grad += out.grad * out_data
    
    out._backward = _backward
    out._prev = {x}
    return out

def log(x: Tensor) -> Tensor:
    """Element-wise natural logarithm with proper gradient flow."""
    # Add small epsilon for numerical stability
    eps = 1e-8
    out_data = np.log(np.maximum(x.data, eps))
    out = Tensor(out_data, requires_grad=x.requires_grad)
    
    def _backward():
        if x.requires_grad:
            # d/dx log(x) = 1/x
            x.grad += out.grad / np.maximum(x.data, eps)
    
    out._backward = _backward
    out._prev = {x}
    return out

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    # Apply softmax to logits - this creates the computation graph
    probs = softmax(logits, dim=-1)
    
    # Convert targets to proper format
    batch_size, num_classes = logits.data.shape
    target_indices = targets.data.astype(int)
    if target_indices.ndim > 1:
        target_indices = target_indices.flatten()
    
    # Create one-hot encoding as a Tensor (requires_grad=False)
    # This is necessary for proper gradient flow through the multiplication
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), target_indices] = 1.0
    one_hot_tensor = Tensor(one_hot, requires_grad=False)
    
    # Compute log probabilities - this maintains gradient flow from probs
    log_probs = log(probs)
    
    # Compute cross entropy: -sum(targets * log(probs)) / batch_size
    # Each operation here maintains the computation graph
    element_wise_loss = one_hot_tensor * log_probs  # Element-wise multiplication
    sum_loss = element_wise_loss.sum()              
    negative_sum = -sum_loss                        
    ce = negative_sum / batch_size                  
    
    return ce