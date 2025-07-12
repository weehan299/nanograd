# Nanograd: A Complete Deep Learning Framework

A minimalist yet comprehensive deep learning framework built from scratch in Python, implementing the core components of PyTorch including automatic differentiation, neural network layers, and a GPT language model.

##  Overview

This project demonstrates a understanding of deep learning fundamentals by implementing a complete neural network framework that implements a GPT architecture from the ground up. The framework includes:

- **Automatic Differentiation Engine**: Custom tensor class with gradient computation
- **Neural Network Layers**: Linear, ReLU, Dropout, Layer Normalization, Multi-Head Attention
- **GPT Architecture**: Complete transformer-based language model implementation
- **Training Infrastructure**: Adam optimizer, loss functions, model state management
- **Memory Optimization**: Efficient gradient computation and graph clearing

## ️ Architecture

### Core Components

#### 1. Tensor (`tensor.py`)
- Custom tensor implementation with automatic differentiation
- Supports broadcasting, matrix multiplication, and element-wise operations
- Memory-efficient backward pass with iterative topological sorting

```python
# Example usage
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.sum()
y.backward()  # Computes gradients automatically
```

#### 2. Neural Network Modules (`module.py`, `nn.py`)
- Base `Module` class for all neural network components
- Automatic parameter registration and gradient management
- Implementation of common layers: Linear, ReLU, Embedding

#### 3. Transformer Components (`multihead_attention.py`, `layernorm.py`)
- Multi-head self-attention mechanism
- Layer normalization with learnable parameters
- Feed-forward networks with residual connections
- Causal attention masking for language modeling

#### 4. GPT Language Model (`gptlanguage.py`)
- Complete GPT architecture implementation
- Configurable model size (embedding dimensions, heads, layers)
- Text generation with temperature sampling
- Proper weight initialization following GPT conventions

##  Key Features

### Automatic Differentiation
- **Forward Mode**: Efficient computation graph construction
- **Backward Mode**: Reverse-mode differentiation with proper gradient flow
- **Memory Management**: Automatic graph clearing to prevent memory leaks

### Optimization
- **Adam Optimizer**: First and second moment estimation with bias correction
- **Gradient Clipping**: Prevents exploding gradients during training
- **Weight Decay**: L2 regularization support

### Model Persistence
- **Save/Load Functionality**: Complete model state serialization
- **Configuration Management**: Automatic hyperparameter saving
- **Cross-session Compatibility**: Resume training from checkpoints

## 🧪 Example Usage

### Training a Simple Model
```python
from src.nn import MLP
from src.optimizer import Adam
from src.tensor import Tensor

# Create model
model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for batch in dataloader:
    x, y = batch
    
    # Forward pass
    logits = model(x)
    loss = cross_entropy(logits, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### GPT Text Generation
```python
from src.gptlanguage import GPTLanguageModel

# Initialize model
model = GPTLanguageModel(
    vocab_size=10000,
    n_embd=384,
    n_head=6,
    n_layer=6,
    block_size=256
)

# Generate text
prompt = encode("Hello world")  # Your tokenization function
generated = model.generate(prompt, max_new_tokens=100)
```

##  Technical Highlights

### Performance Optimizations
- **Efficient Broadcasting**: Proper gradient handling for broadcasted operations
- **Memory Management**: Iterative backward pass prevents stack overflow
- **Numerical Stability**: Proper handling of softmax and log operations

### Educational Value
- **Clean Implementation**: Easy-to-understand code structure
- **Comprehensive Documentation**: Detailed comments explaining each component
- **Modular Design**: Each component can be studied independently

## ️ Project Structure

```
src/
├── tensor.py              # Core tensor with autograd
├── module.py              # Base neural network module
├── nn.py                  # Basic neural network layers
├── optimizer.py           # Adam optimizer implementation
├── softmax.py             # Softmax and cross-entropy functions
├── dropout.py             # Dropout regularization
├── layernorm.py           # Layer normalization
├── multihead_attention.py # Multi-head attention mechanism
├── gptlanguage.py         # Complete GPT implementation
└── model_state.py         # Model saving/loading utilities
```


##  Getting Started

1. **Clone the repository**
2. **Install dependencies**: `numpy` (only external dependency!)
3. **Run examples**: See individual module files for usage examples
4. **Experiment**: Modify architectures and hyperparameters

##  Technical Details

### Gradient Computation
The framework implements reverse-mode automatic differentiation with:
- Dynamic computation graph construction
- Efficient memory usage through graph clearing
- Proper handling of broadcasting and tensor operations

### Attention Mechanism
Multi-head attention implementation includes:
- Scaled dot-product attention
- Causal masking for autoregressive generation
- Dropout for regularization
- Residual connections and layer normalization

## 📈 Future Enhancements

Potential extensions include:
- Additional optimizers (SGD, RMSprop)
- Convolutional layers for computer vision
- Batch normalization
- Advanced regularization techniques
- GPU acceleration support

