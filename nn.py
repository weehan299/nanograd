import numpy as np
from module import Module
from tensor import Tensor

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(
            # Kaiming initialisation
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return x.matmul(self.weight) + self.bias

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        data = np.maximum(0,x.data)
        out = Tensor(data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        out._prev = {x}
        return out



def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    return (diff * diff).sum()



class MLP(Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1   = Linear(in_dim,  hidden_dim)
        self.act1  = ReLU()
        self.l2   = Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        return x

class Embedding(Module):
    """
    Embedding layer that maps discrete tokens to dense vectors.
    Similar to nn.Embedding in PyTorch.
    """
    def __init__(self, num_embeddings, embedding_dim):

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embedding weights with small random values
        # Using Xavier/Glorot initialization
        std = np.sqrt(2.0 / (num_embeddings + embedding_dim))
        self.weight = Tensor(
            np.random.normal(0, std, (num_embeddings, embedding_dim)),
            requires_grad=True
        )
    
    def forward(self, input_ids: Tensor) -> Tensor:

        # Convert input to numpy for indexing
        indices = input_ids.data.astype(int)
        
        # Check bounds
        if np.any(indices < 0) or np.any(indices >= self.num_embeddings):
            raise ValueError(f"Input indices must be in range [0, {self.num_embeddings})")
        
        # Get embeddings by indexing into weight matrix
        output_data = self.weight.data[indices]
        
        out = Tensor(output_data, requires_grad=self.weight.requires_grad)
        
        def _backward():
            if self.weight.requires_grad and out.grad is not None:
                # Accumulate gradients for each embedding
                # We need to handle the case where the same index appears multiple times
                grad_weight = np.zeros_like(self.weight.data)
                
                # Flatten indices and gradients for easier processing
                flat_indices = indices.flatten()
                flat_grad = out.grad.reshape(-1, self.embedding_dim)
                
                # Accumulate gradients
                for i, idx in enumerate(flat_indices):
                    grad_weight[idx] += flat_grad[i]
                
                self.weight.grad += grad_weight
        
        out._backward = _backward
        out._prev = {input_ids, self.weight}
        return out

# Example usage and test
if __name__ == "__main__":
    print("Testing Embedding layer:")
    
    # Test 1: Basic embedding
    vocab_size = 10
    embed_dim = 4
    embedding = Embedding(vocab_size, embed_dim)
    
    # Create some token indices
    tokens = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=False)
    print(f"Input tokens shape: {tokens.data.shape}")
    print(f"Input tokens:\n{tokens.data}")
    
    # Get embeddings
    embeddings = embedding(tokens)
    print(f"Embeddings shape: {embeddings.data.shape}")
    print(f"Embeddings:\n{embeddings.data}")
    
    # Test backward pass
    loss = embeddings.sum()
    loss.backward()
    print(f"Embedding weight gradient shape: {embedding.weight.grad.shape}")
    print(f"Embedding weight gradient sum: {embedding.weight.grad.sum()}")
    
    print("\n" + "="*50)
    print("Testing PositionalEncoding layer:")
    

