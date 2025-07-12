import numpy as np
from src.module import Module
from src.tensor import Tensor
from src.nn import Linear, Embedding, PositionalEncoding
from src.layernorm import LayerNorm
from src.multihead_attention import Block
from src.softmax import cross_entropy, softmax
from src.dropout import Dropout


class GPTLanguageModel(Module):
    """GPT Language Model implementation"""
    
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        
        # Token embeddings
        self.token_embedding_table = Embedding(vocab_size, n_embd)
        # Positional embeddings
        self.position_embedding_table = PositionalEncoding(block_size, n_embd)
        # Transformer blocks
        self.blocks = []
        for i in range(n_layer):
            block = Block(n_embd, n_head, block_size, dropout)
            self.blocks.append(block)
            setattr(self, f'block_{i}', block)
        # Final layer norm
        self.ln_f = LayerNorm(n_embd)
        # Language model head
        self.lm_head = Linear(n_embd, vocab_size)
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights similar to GPT"""
        # Token embeddings
        std = 0.02
        self.token_embedding_table.weight.data = np.random.normal(0, std, self.token_embedding_table.weight.data.shape)
        # Positional embeddings
        self.position_embedding_table.embedding.weight.data = np.random.normal(0, std, self.position_embedding_table.embedding.weight.data.shape)
        # Initialize all linear layers
        for param in self.parameters():
            if len(param.data.shape) == 2:  # Weight matrix
                param.data = np.random.normal(0, std, param.data.shape)
            elif len(param.data.shape) == 1:  # Bias vector
                param.data = np.zeros(param.data.shape)
    
    def forward(self, idx: Tensor, targets=None):
        """Forward pass through the model"""
        B, T = idx.data.shape
        # Token embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        # Positional embeddings
        pos_emb = self.position_embedding_table(T)  # (T, n_embd)
        # Add token and positional embeddings
        x = tok_emb + pos_emb  # Broadcasting: (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        # Final layer norm
        x = self.ln_f(x)
        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            # Reshape for cross-entropy loss
            B, T, C = logits.data.shape
            logits_flat = logits.reshape((B * T, C))
            targets_flat = targets.reshape((B * T,))
            loss = cross_entropy(logits_flat, targets_flat)
        return logits, loss

    
    def generate(self, idx: Tensor, max_new_tokens: int):
        """Generate new tokens autoregressively"""
        # idx is (B, T) array of indices in the current context
        generated = []
        
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = self.crop_sequence(idx, self.block_size)
            
            # Get predictions
            logits, _ = self.forward(idx_cond)
            
            # Focus only on the last time step
            logits = logits.data[:, -1, :]  # (B, vocab_size)
            
            # Apply softmax to get probabilities
            probs = softmax(Tensor(logits,requires_grad=False), dim=-1)
            
            # Sample from the distribution
            idx_next = self.multinomial_sample(probs)
            # Append sampled index to the running sequence
            idx = self.concat_sequences(idx, idx_next)
            
        return idx.data
    
    def crop_sequence(self, tensor: Tensor, max_len: int):
        if tensor.data.shape[1] <= max_len:
            return tensor
        # Take the last max_len tokens
        out_data = tensor.data[:, -max_len:]
        out = Tensor(out_data, requires_grad=tensor.requires_grad)
        
        return out
    
    def multinomial_sample(self, probs: Tensor):
        """Sample from multinomial distribution"""
        # Convert to numpy for sampling
        probs_np = probs.data[0]  # Take first batch
        # Sample
        sample_idx = np.random.choice(len(probs_np), p=probs_np)
        
        # Return as tensor
        return Tensor([[sample_idx]], requires_grad=False)
    
    def concat_sequences(self, seq1: Tensor, seq2: Tensor):
        """Concatenate two sequences along time dimension"""
        out_data = np.concatenate([seq1.data, seq2.data], axis=1)
        out = Tensor(out_data, requires_grad=seq1.requires_grad or seq2.requires_grad)
        return out
