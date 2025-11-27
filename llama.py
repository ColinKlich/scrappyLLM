"""
A PyTorch-based implementation of a Llama-like language model for character-level text generation.
This script covers the entire pipeline from data loading and preprocessing to model definition,
training, and text generation.
"""

# --- Imports ---
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from collections import OrderedDict

# --- Device Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Configuration ---
MASTER_CONFIG = {
    'batch_size': 32,          # Number of batches to be processed at each random split
    'context_window': 64,    # Number of characters in each input (x) and target (y) sequence of each batch
    'd_model': 128,
    'epochs': 10000,         # Number of training epochs
    'log_interval': 10,      # Log information every 10 batches during training
    'n_heads': 8,
    'n_layers': 4
}

# --- Data Loading & Preprocessing ---
lines = open("./data/tinyshakespeare.txt", 'r').read()

# Create a sorted list of unique characters in the dataset
vocab = sorted(list(set(lines)))
MASTER_CONFIG['vocab_size'] = len(vocab)

# Mapping integers to characters (itos)
itos = {i: ch for i, ch in enumerate(vocab)}

# Mapping characters to integers (stoi)
stoi = {ch: i for i, ch in enumerate(vocab)}


def encode(s):
    """
    Converts a string to a list of integers using the character-to-integer mapping.
    Args:
        s (str): The input string.
    Returns:
        list[int]: The list of encoded integers.
    """
    return [stoi[ch] for ch in s]

def decode(l):
    """
    Converts a list of integers back to a string using the integer-to-character mapping.
    Args:
        l (list[int]): The list of integers.
    Returns:
        str: The decoded string.
    """
    return ''.join([itos[i] for i in l])

# Convert the dataset into a torch tensor with specified data type (dtype)
dataset = torch.tensor(encode(lines), dtype=torch.int8)

# Display the shape of the resulting tensor
print(dataset.shape)


# --- Data Batching ---
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    """
    Generates batches of data for training, validation, or testing.
    Args:
        data (torch.Tensor): The full dataset.
        split (str): 'train', 'val', or 'test'.
        batch_size (int): The number of sequences in a batch.
        context_window (int): The length of each sequence.
        config (dict): The master configuration dictionary.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing input sequences (x) and target sequences (y).
    """
    # Split the dataset into training, validation, and test sets
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    # Determine which split to use
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # Pick random starting points within the data
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))

    # Create input sequences (x) and corresponding target sequences (y)
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    # Move batches to the selected device
    x, y = x.to(device), y.to(device)

    return x, y

# --- Training & Evaluation ---
@torch.no_grad()  # Don't compute gradients for this function
def evaluate_loss(model, config=MASTER_CONFIG):
    """
    Evaluates the model's loss on the training and validation sets.
    Args:
        model (nn.Module): The model to evaluate.
        config (dict): The master configuration dictionary.
    Returns:
        dict: A dictionary containing the mean loss for 'train' and 'val' splits.
    """
    # Placeholder for the evaluation results
    out = {}
    
    # Set the model to evaluation mode
    model.eval()

    # Iterate through training and validation splits
    for split in ["train", "val"]:
        # Placeholder for individual losses
        losses = []

        # Generate 10 batches for evaluation
        for _ in range(10):
            # Get input sequences (xb) and target sequences (yb)
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            
            # Perform model inference and calculate the loss
            _, loss = model(xb, yb)
            
            # Append the loss to the list
            losses.append(loss.item())

        # Calculate the mean loss for the split and store it in the output dictionary
        out[split] = np.mean(losses)
    
    # Set the model back to training mode
    model.train()
    
    return out

def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    """
    Trains the language model.
    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        config (dict): The master configuration dictionary.
        print_logs (bool): Whether to print training logs. Defaults to False.
    Returns:
        matplotlib.axes.Axes: A plot of the training and validation losses.
    """
    # Placeholder for storing losses
    losses = []
    
    # Start tracking time
    start_time = time.time()

    # Iterate through epochs
    for epoch in range(config['epochs']):
        # Zero out gradients
        optimizer.zero_grad()

        # Obtain batches for training
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])

        # Forward pass through the model to calculate logits and loss
        logits, loss = model(xs, targets=ys)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # If a learning rate scheduler is provided, adjust the learning rate
        if scheduler:
            scheduler.step()

        # Log progress every specified interval
        if epoch % config['log_interval'] == 0:
            # Calculate batch time
            batch_time = time.time() - start_time
            
            # Evaluate loss on validation set
            x = evaluate_loss(model)
            
            # Store the validation loss
            losses.append(x)
            
            # Print progress logs if specified
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
                
            # Reset the timer
            start_time = time.time()

            # Print learning rate if a scheduler is provided
            if scheduler:
                print(f"lr: {scheduler.get_lr()}")

    # Print the final validation loss
    print(f"Validation loss: {losses[-1]['val']:.4f}")
    
    # Plot the training and validation loss curves
    return pd.DataFrame(losses).plot()

# --- Model Components ---
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Args:
        layer_shape (tuple): The shape of the layer to be normalized.
        eps (float): A small value added to the denominator for numerical stability.
        bias (bool): Not used in this implementation.
    """
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super().__init__()
        # Registering a learnable parameter 'scale' as a parameter of the module
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))
        self.eps = eps

    def forward(self, x):
        """
        Assumes shape is (batch, seq_len, d_model)
        """
        # Calculating the Frobenius norm, RMS = 1/sqrt(N) * Frobenius norm
        norm_x = x.norm(2, dim=(1,2), keepdim=True)
        rms_x = norm_x * (x.shape[1] * x.shape[2])**-0.5

        # Normalizing the input tensor 'x' with respect to RMS
        x_normed = x / (rms_x + self.eps)

        # Scaling the normalized tensor using the learnable parameter 'scale'
        return self.scale * x_normed

class RoPEMaskedAttentionHead(nn.Module):
    """
    An attention head with Rotary Position Embeddings (RoPE) and causal masking.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)

        self.register_buffer('R', self.get_rotary_matrix(config['context_window'], config['d_model']))

    @staticmethod
    def get_rotary_matrix(context_window, embedding_dim):
        """
        Generates the rotary matrix for positional embeddings.
        """
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = -np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R
    
    def forward(self, x, return_attn_weights=False):
        b, m, d = x.shape
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)

        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=.1, is_causal=True
        )

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m,m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations
    
class RoPEMaskedMultiheadAttention(nn.Module):
    """
    Multi-head attention module with RoPE and causal masking.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Create a list of RoPEMaskedAttentionHead instances as attention heads
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])  # Linear layer after concatenating heads
        self.dropout = nn.Dropout(.1)  # Dropout layer

    def forward(self, x):
        # Process each attention head and concatenate the results
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        
        # Apply linear transformation and dropout
        x = self.linear(x)
        x = self.dropout(x)
        return x

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit activation function.
    Reference: https://arxiv.org/pdf/2002.05202v1.pdf
    """
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out

class LlamaBlock(nn.Module):
    """
    A single block of the Llama model, containing attention and feed-forward layers.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rms = RMSNorm((config['context_window'], config['d_model']))
        
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        # One block of attention with pre-normalization and residual connection
        x_norm = self.rms(x)
        x = x + self.attention(x_norm)

        # Feed-forward block with pre-normalization and residual connection
        x_norm = self.rms(x)
        x = x + self.feedforward(x_norm)
        return x

# --- Model Definition ---
class LlamaModel(nn.Module):
    """
    The main Llama-like model architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layer for input tokens
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # A sequence of LlamaBlocks
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )

        # Final linear layer for prediction
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        
        x = self.llama_blocks(x)

        # Final linear layer to get logits
        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

# --- Text Generation ---
def generate(model, config=MASTER_CONFIG, max_new_tokens=30):
    """
    Generates text using the trained model.
    Args:
        model (nn.Module): The trained model.
        config (dict): The master configuration dictionary.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        list[str]: A list of generated text samples.
    """
    # Start with a tensor of zeros on the correct device
    idx = torch.zeros(5, 1).long().to(device)
    for _ in range(max_new_tokens):
        # Crop idx to the last context_window tokens
        logits = model(idx[:, -config['context_window']:])
        # Get the logits for the last time step
        last_time_step_logits = logits[:, -1, :]
        
        # Apply softmax to get probabilities
        p = F.softmax(last_time_step_logits, dim=-1)
        # Sample from the distribution to get the next token
        idx_next = torch.multinomial(p, num_samples=1)
        # Append the new token to the sequence
        idx = torch.cat([idx, idx_next], dim=-1)
    return [decode(x) for x in idx.tolist()]


# --- Main Execution ---
if __name__ == '__main__':
    # Create an instance of the LlamaModel
    model = LlamaModel(MASTER_CONFIG)
    
    # Move the model to the selected device
    model.to(device)

    # Define the Adam optimizer for model parameters
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    train(model, optimizer, print_logs=True)

    # Generate text using the trained model
    print(generate(model))