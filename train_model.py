import numpy as np
from src.tensor import Tensor
from src.gptlanguage import GPTLanguageModel
from src.optimizer import Adam
from src.model_state import save_model, load_model, model_exists
import os


# Set numpy to use all CPU cores
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())


# OPTIMIZED HYPERPARAMETERS FOR CPU TRAINING
batch_size = 16          # Reduced from 64
block_size = 128         # Reduced from 256
max_iters = 5000        # Reduced from 5000
eval_interval = 200     # Reduced from 500
learning_rate = 1e-3    # Increased from 3e-4
eval_iters = 50         # Reduced from 200
n_embd = 256            # Reduced from 384
n_head = 4              # Reduced from 6
n_layer = 6             # Reduced from 6
dropout = 0.1           # Reduced from 0.2
# new schedule params
base_lr       = learning_rate         # peak LR after warmup
warmup_iters  = 500                   # 10% of max_iters


# Model save/load settings
model_save_path = "models/gpt_model.json"
save_interval = 1000  # Save model every 1000 iterations

# Set random seed for reproducibility
np.random.seed(1337)

try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    # If file doesn't exist, create a simple dummy dataset
    print(f"Warning: 'input.txt' not found. Creating dummy dataset.")
    text = "Hello world! This is a simple text for testing the GPT model. " * 1000

# Create character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# Encoder and decoder functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
    
# Encode the entire text
data = np.array(encode(text), dtype=np.int32)

# Train/validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Total data length: {len(data)}")
print(f"Training data length: {len(train_data)}")
print(f"Validation data length: {len(val_data)}")

def get_batch(train_data, val_data, split, batch_size=4, block_size=8):
    """Generate a batch of data"""
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)

def estimate_loss(model, train_data, val_data, eval_iters=10, batch_size=4, block_size=8):
    """Estimate loss on train and validation sets"""
    model.training = False  # Set to evaluation mode
    out = {}
    
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split, batch_size, block_size)
            logits, loss = model(X, Y)
            losses.append(loss.data.item())
        out[split] = np.mean(losses)
    
    model.training = True  # Set back to training mode
    return out

def train_gpt():
    """Main training function with save/load functionality"""
    
    # Check if model exists and load it
    if model_exists(model_save_path):
        print(f"Loading existing model from {model_save_path}")
        model, additional_info = load_model(model_save_path)
        
        # Get the last iteration from additional info
        start_iter = additional_info.get('last_iteration', 0)
        best_val_loss = additional_info.get('best_val_loss', float('inf'))
        
        print(f"Resuming training from iteration {start_iter}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")
    else:
        print("Creating new model")
        # Create model
        model = GPTLanguageModel(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
            dropout=dropout
        )
        start_iter = 0
        best_val_loss = float('inf')
    
    print(f"Model parameters: {sum(p.data.size for p in model.parameters()) / 1e6:.2f}M")
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for iter in range(start_iter, max_iters):
        # Evaluate loss periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Save model if it's the best so far
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                additional_info = {
                    'last_iteration': iter,
                    'best_val_loss': best_val_loss,
                    'train_loss': losses['train'],
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'block_size': block_size
                }
                save_model(model, model_save_path.replace('.json', '_best.json'), additional_info)
                print(f"New best model saved! Validation loss: {best_val_loss:.4f}")
        
        # Save model periodically
        if iter % save_interval == 0 and iter > 0:
            additional_info = {
                'last_iteration': iter,
                'best_val_loss': best_val_loss,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'block_size': block_size
            }
            save_model(model, model_save_path, additional_info)
            print(f"Model checkpoint saved at iteration {iter}")
        
        # Sample a batch of data
        xb, yb = get_batch(train_data, val_data, 'train', batch_size, block_size)
        
        
        # Forward pass
        logits, loss = model(xb, yb)
        # Linear decay of learning rates
        if iter < warmup_iters:
            lr = base_lr * (iter / warmup_iters)
        else:
            lr = base_lr * (1 - (iter / max_iters))
        optimizer.lr = lr
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every 100 iterations
        if iter % 100 == 0:
            print(f"iter {iter}: loss {loss.data:.4f}")
    
    # Save final model
    additional_info = {
        'last_iteration': max_iters,
        'best_val_loss': best_val_loss,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'block_size': block_size,
        'training_completed': True
    }
    save_model(model, model_save_path, additional_info)
    print("Final model saved!")
    
    # Generate some text
    print("\nGenerating text...")
    context = Tensor([[encode("Hello")[0]]], requires_grad=False)  # Start with "Hello"
    generated_idx = model.generate(context, max_new_tokens=100)
    
    # Decode the generated text
    full_sequence = generated_idx.data.tolist()[0]
    generated_text = decode(full_sequence)
    print(f"Generated text: {generated_text}")
    
    return model

def test_save_load():
    """Test the save/load functionality"""
    print("Testing save/load functionality...")
    
    # Create a small model
    model = GPTLanguageModel(
        vocab_size=50,
        n_embd=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        dropout=0.1
    )
    
    # Save model
    test_path = "test_model.json"
    additional_info = {'test': True, 'version': 1.0}
    save_model(model, test_path, additional_info)
    
    # Load model
    loaded_model, loaded_info = load_model(test_path)
    
    # Test that weights are the same
    original_params = list(model.parameters())
    loaded_params = list(loaded_model.parameters())
    
    for i, (orig, loaded) in enumerate(zip(original_params, loaded_params)):
        if not np.allclose(orig.data, loaded.data):
            print(f"Error: Parameter {i} does not match!")
            return False
    
    print("Save/load test passed!")
    print(f"Additional info loaded: {loaded_info}")
    
    # Clean up
    import os
    if os.path.exists(test_path):
        os.remove(test_path)
    
    return True

def generate_from_saved_model(model_path: str = None, prompt: str = "Hello", max_tokens: int = 1000):
    """Generate text from a saved model"""
    if model_path is None:
        model_path = model_save_path
    
    if not model_exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    model, additional_info = load_model(model_path)
    
    print(f"Model info: {additional_info}")
    
    # Generate text
    try:
        context = Tensor([[encode(prompt)[0]]], requires_grad=False)
        generated_idx = model.generate(context, max_new_tokens=max_tokens)
        
        # Decode the generated text
        full_sequence = generated_idx.data.tolist()[0]
        generated_text = decode(full_sequence)
        print(f"Generated text: {generated_text}")
        
    except Exception as e:
        print(f"Error generating text: {e}")

if __name__ == "__main__":
    print("Testing save/load functionality...")
    #test_save_load()
    
    print("\n" + "="*50)
    print("Starting training with save/load...")
    
    # Run training
    model = train_gpt()
    
    print("\n" + "="*50)
    print("Testing text generation from saved model...")
    generate_from_saved_model()
    
    print("Training completed!")