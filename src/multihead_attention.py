import numpy as np
from src.module import Module
from src.tensor import Tensor
from src.nn import Linear, ReLU
from src.dropout import Dropout
from src.softmax import softmax
from src.layernorm import LayerNorm


class Head(Module):
    """One head of self-attention"""
    
    def __init__(self, head_size, n_embd, block_size, dropout_p=0.1):
        super().__init__()
        self.head_size = head_size
        self.n_embd = n_embd
        self.block_size = block_size
        
        self.key = Linear(n_embd, head_size, bias=False)
        self.query = Linear(n_embd, head_size, bias=False)
        self.value = Linear(n_embd, head_size, bias=False)
        # Create lower triangular mask for causal attention
        self.tril = np.tril(np.ones((block_size, block_size)))
        
        self.dropout = Dropout(dropout_p)
    
    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (batch_size, seq_len, n_embd)
        B, T, C = x.data.shape
        # Compute key, query, value projections
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        # Compute attention scores
        # q @ k.T -> (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q.matmul(k.transpose()) * (self.head_size ** -0.5)
        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Apply softmax
        wei = softmax(wei, dim=-1)
        # Apply dropout
        wei = self.dropout(wei)
        # Weighted aggregation of values
        out = wei.matmul(v)  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
    


class MultiHeadAttention(Module):
    """Multiple heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout_p=0.1):
        super().__init__()
        self.heads = [Head(head_size, n_embd, block_size, dropout_p) for _ in range(num_heads)]
        # Register heads as submodules
        for i, head in enumerate(self.heads):
            setattr(self, f'head_{i}', head)
        self.proj = Linear(head_size * num_heads, n_embd)
        self.dropout = Dropout(dropout_p)
    
    def forward(self, x: Tensor) -> Tensor:
        # Run all heads in parallel and concatenate
        head_outputs = [head(x) for head in self.heads]
        # Concatenate along the last dimension
        out = self.concat(head_outputs, dim=-1)
        # Apply projection
        out = self.proj(out)
        # Apply dropout
        out = self.dropout(out)
        return out
    
    def concat(self, tensors, dim=-1):
        """Concatenate tensors along specified dimension"""
        if not tensors:
            raise ValueError("Cannot concatenate empty list of tensors")
        
        # Handle negative dimension
        if dim < 0:
            dim = tensors[0].data.ndim + dim
        
        # Concatenate data
        data_list = [t.data for t in tensors]
        out_data = np.concatenate(data_list, axis=dim)
        
        # Determine if gradient is required
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(out_data, requires_grad=requires_grad)
        
        def _backward():
            if out.grad is not None:
                # Split gradient back to original tensors
                split_indices = []
                current_idx = 0
                for t in tensors:
                    size = t.data.shape[dim]
                    split_indices.append(current_idx + size)
                    current_idx += size
                
                # Remove the last index as it's just the total size
                split_indices = split_indices[:-1]
                
                # Split the gradient
                grad_splits = np.split(out.grad, split_indices, axis=dim)
                
                # Add gradients to input tensors
                for i, t in enumerate(tensors):
                    if t.requires_grad:
                        t.grad += grad_splits[i]
        
        out._backward = _backward
        out._prev = set(tensors)
        return out


class FeedForward(Module):
    """Feed-forward network with ReLU activation"""
    
    def __init__(self, n_embd, dropout_p=0.1):
        super().__init__()
        self.net = [
            Linear(n_embd, 4 * n_embd),
            ReLU(),
            Linear(4 * n_embd, n_embd),
            Dropout(dropout_p)
        ]
        
        # Register as submodules
        for i, layer in enumerate(self.net):
            setattr(self, f'layer_{i}', layer)
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.net:
            x = layer(x)
        return x


class Block(Module):
    """Transformer block: communication followed by computation"""
    
    def __init__(self, n_embd, n_head, block_size, dropout_p=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout_p)
        self.ffwd = FeedForward(n_embd, dropout_p)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
    
    def forward(self, x: Tensor) -> Tensor:
        # Self-attention with residual connection
        x = x + self.sa(self.ln1(x))
        # Feed-forward with residual connection
        x = x + self.ffwd(self.ln2(x))
        return x
