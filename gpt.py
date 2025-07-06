

# Text processing code converted to work with custom tensor framework
import numpy as np
from tensor import Tensor
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Load the text data
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = np.array(encode(text), dtype=np.int32)  # Using numpy instead of torch.tensor
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

print(f"Total data length: {len(data)}")
print(f"Training data length: {len(train_data)}")
print(f"Validation data length: {len(val_data)}")

# Hyperparameters (you'll need to define these based on your model)
block_size = 8  # context length
batch_size = 4  # batch size

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)