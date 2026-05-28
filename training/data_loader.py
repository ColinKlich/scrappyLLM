"""
Data loading utilities for Scrappy LLM.
Adapted from pirate_llm-main training/data.py

Supports both character-level and BPE tokenization with memmap-based
efficient loading of large datasets.
"""

import numpy as np
import torch
from pathlib import Path


def load_char_data(data_path: str):
    """Load character-level data from text file.
    
    Args:
        data_path: Path to text file
        
    Returns:
        tuple: (encoded_data as list, config with vocab_size)
    """
    with open(data_path, 'r') as f:
        lines = f.read()
    
    vocab = sorted(list(set(lines)))
    vocab_size = len(vocab)
    
    itos = {i: ch for i, ch in enumerate(vocab)}
    stoi = {ch: i for i, ch in enumerate(vocab)}
    
    def encode(s):
        return [stoi[ch] for ch in s]
    
    encoded = encode(lines)
    return encoded, {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi, 'data': encoded}


def encode_data_to_bin(data: list[int], train_path: str, val_path: str, test_size: float = 0.1):
    """Encode dataset and save to binary .bin files.
    
    Args:
        data: List of token IDs
        train_path: Path for training binary file
        val_path: Path for validation binary file
        test_size: Proportion for test split (ignored if not used)
    """
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    
    train_data = np.array(data[:train_size], dtype=np.uint16)
    val_data = np.array(data[train_size:train_size + val_size], dtype=np.uint16)
    
    np.savetxt(train_path, train_data, fmt='%u')
    np.savetxt(val_path, val_data, fmt='%u')
    
    print(f"Saved training data to {train_path} ({len(train_data)} tokens)")
    print(f"Saved validation data to {val_path} ({len(val_data)} tokens)")
    return len(train_data), len(val_data)


def get_batch(data_type, config, split='train'):
    """Sample a random batch from train.bin or val.bin.
    
    Args:
        data_type: 'char' for character tensor or 'memmap' for binary files
        config: Training configuration
        split: 'train' or 'val'
        
    Returns:
        x: input sequence tensor of shape (batch_size, context_window)
        y: target sequence tensor (shifted by 1)
    """
    if data_type == 'memmap':
        if split == 'train':
            data = np.memmap(config.train_bin, dtype=np.uint16, mode='r')
        elif split == 'val':
            data = np.memmap(config.val_bin, dtype=np.uint16, mode='r')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        context_window = getattr(config, 'context_window', getattr(config, 'block_size', 128))
        batch_size = getattr(config, 'batch_size', 32)

        ix = torch.randint(len(data) - context_window - 1, (batch_size,))

        x = torch.stack([torch.from_numpy(data[i:i+context_window].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+context_window+1].astype(np.int64)) for i in ix])
        
        device = getattr(config, 'device', 'cpu')
        if device == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        
        return x, y
    
    elif data_type == 'char':
        if split == 'train':
            data = config.data['train']
        elif split == 'val':
            data = config.data['val']
        
        context_window = getattr(config, 'context_window', getattr(config, 'block_size', 128))
        batch_size = getattr(config, 'batch_size', 32)
        device = getattr(config, 'device', 'cpu')

        ix = torch.randint(0, len(data) - context_window - 1, (batch_size,))

        x = torch.stack([data[i:i+context_window] for i in ix])
        y = torch.stack([data[i+1:i+context_window+1] for i in ix])

        x, y = x.long().to(device), y.long().to(device)
        return x, y
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def load_or_create_data(data_path: str, config):
    """Load or load character data based on config."""
    if hasattr(config, 'data_type') and (config.data_type == 'memmap' or (hasattr(config, 'train_bin') and Path(config.train_bin).exists())):
        print(f"Loading from memmap binaries: {config.train_bin}")
        data = {'train_bin': config.train_bin, 'val_bin': config.val_bin}
    else:
        print(f"Loading character data from {data_path}")
        encoded, vocab_info = load_char_data(data_path)
        data = {
            'train': encoded[:int(0.8 * len(encoded))],
            'val': encoded[int(0.8 * len(encoded)):int(0.9 * len(encoded))],
            'data': vocab_info['data'],
            'vocab_size': vocab_info['vocab_size']
        }
        config.vocab_size = vocab_info['vocab_size']
    
    return data, config
