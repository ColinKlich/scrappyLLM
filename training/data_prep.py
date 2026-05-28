"""
Training data and tokenization support for Scrappy LLM.
Adapted from pirate_llm-main dataset utilities.
"""

import numpy as np
import os
from pathlib import Path


def train_char_vocab(text: str) -> tuple[dict, dict, dict]:
    """Train character vocabulary from text.
    
    Args:
        text: Raw text corpus
        
    Returns:
        stoi: string-to-int mapping
        itos: int-to-string mapping
        vocab: dict of {str(i): char}
    """
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    vocab_dict = {str(i): ch for i, ch in enumerate(vocab)}
    
    return stoi, itos, vocab_dict


def create_bin_from_text(
    text: str,
    bin_path: str,
    stoi: dict,
    special_tokens=None,
    context_window: int = 128,
    split_ratios: tuple = (0.8, 0.1, 0.1)
):
    """Create binary .bin files from text for efficient loading.
    
    Args:
        text: Raw text corpus
        bin_path: Base path for output files
        stoi: Token-to-ID mapping
        special_tokens: Optional special token definitions
        context_window: Token sequence length
        split_ratios: (train, val, test) ratios
        
    Returns:
        train_tokens, val_tokens: Number of tokens in each split
    """
    # Encode text
    encoded = [stoi.get(ch, 0) for ch in text]
    
    # Split
    train_end = int(split_ratios[0] * len(encoded))
    val_end = train_end + int(split_ratios[1] * len(encoded))
    
    train_data = np.array(encoded[:train_end], dtype=np.uint16)
    val_data = np.array(encoded[train_end:val_end], dtype=np.uint16)
    
    # Save
    np.save(bin_path.replace('.bin', '_train.npy'), train_data)
    np.save(bin_path.replace('.bin', '_val.npy'), val_data)
    
    print(f"Saved training data: {len(train_data):,} tokens to {bin_path.replace('.bin', '_train.npy')}")
    print(f"Saved validation data: {len(val_data):,} tokens to {bin_path.replace('.bin', '_val.npy')}")
    
    return len(train_data), len(val_data)


def encode_to_uint16(tokens: list[int], filepath: str):
    """Encode token IDs to uint16 numpy array.
    
    Args:
        tokens: List of token IDs
        filepath: Output filepath
    """
    arr = np.array(tokens, dtype=np.uint16)
    np.save(filepath, arr)
    print(f"Saved {len(arr):,} tokens to {filepath}")


def load_tokenized_data(bin_file: str):
    """Load tokenized data from .bin or .npy file.
    
    Args:
        bin_file: Path to token file
        
    Returns:
        Data as numpy array
    """
    if bin_file.endswith('.bin'):
        return np.fromfile(bin_file, dtype=np.uint16)
    else:
        return np.load(bin_file)


def prepare_dataset(
    data_path: str,
    output_dir: str = './data',
    vocab_size: int = 8192,
    context_window: int = 128,
    use_bpe: bool = False
):
    """Prepare training dataset.
    
    Args:
        data_path: Path to training text file
        output_dir: Output directory for processed data
        vocab_size: Target vocabulary size
        context_window: Token sequence length
        use_bpe: Whether to use BPE tokenization
        
    Returns:
        config: Configuration dict with vocab info
    """
    os.makedirs(output_dir, exist_ok=True)
    output_bin = os.path.join(output_dir, 'train.bin')
    
    # Load text
    with open(data_path, 'r') as f:
        text = f.read()
    
    print(f"Loaded {len(text):,} characters from {data_path}")
    
    if use_bpe:
        # BPE tokenization (use tokenizer.py)
        from tokenizer import Tokenizer
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)
        vocab_size = min(tokenizer.vocab_size, vocab_size)
        print(f"Using BPE tokenizer: {tokenizer.vocab_size} tokens")
    else:
        # Character-level vocabulary
        print("Building character vocabulary...")
        stoi, itos, vocab = train_char_vocab(text)
        vocab_size = len(stoi)
        token_ids = [stoi[ch] for ch in text]
        print(f"Character vocabulary: {vocab_size} unique characters")
    
    # Create binary files
    train_tokens, val_tokens = create_bin_from_text(
        text, output_bin, stoi,
        context_window=context_window
    )
    
    # Save vocab
    vocab_path = os.path.join(output_dir, 'vocab.json')
    import json
    with open(vocab_path, 'w') as f:
        json.dump({str(k): v for k, v in vocab.items()}, f, indent=2)
    
    config = {
        'vocab_size': vocab_size,
        'context_window': context_window,
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'data_path': data_path,
        'vocab_path': vocab_path
    }
    
    return config, vocab, stoi, itos
