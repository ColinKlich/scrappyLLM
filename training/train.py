"""
Updated training.py with improved tokenization, training, and generation.
Integrates improvements from pirate_llm-main while preserving your model design.
"""

import math
import os
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import argparse

# Local imports
from .config import Config
from .tokenizer import Tokenizer, CharTokenizer
from .data_loader import get_batch
from .sft_trainer import sft_train, get_sft_batch


# --- Device Configuration ---
def setup_device(config):
    """Set up training device and configuration."""
    if config.device == 'auto':
        if torch.cuda.is_available():
            config.device = 'cuda'
        elif torch.backends.mps.is_available():
            config.device = 'mps'
        else:
            config.device = 'cpu'
    print(f"Using device: {config.device}")
    return config


# --- Training Configuration ---
def build_optimizer(model: nn.Module, config) -> torch.optim.AdamW:
    """Build AdamW optimizer with weight decay on norm params."""
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: AdamW, lr={config.learning_rate}, "
          f"betas=({config.beta1}, {config.beta2})")
    print(f"  Decayed params: {n_decay:,} ({len(decay_params)} tensors)")
    print(f"  No-decay params: {n_no_decay:,} ({len(no_decay_params)} tensors)")
    
    return optimizer


def get_training_lr(it: int, config) -> float:
    """Warmup + cosine decay learning rate schedule."""
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# --- Evaluation ---
@torch.no_grad()
def estimate_loss(model: nn.Module, config):
    """Evaluate model on train/val splits."""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            # Use memmap mode if binary files exist
            data_path = getattr(config, 'train_bin', None) if split == 'train' else getattr(config, 'val_bin', None)
            if data_path and os.path.exists(data_path):
                x, y = get_batch('memmap', config, split)
            else:
                x, y = get_batch('char', config, split)
            _, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()
    
    model.train()
    return out


# --- Checkpoint Handling ---
def try_resume(config, out_dir: str) -> Optional[dict]:
    """Try to resume from local checkpoint."""
    local_path = os.path.join(out_dir, 'ckpt.pt')
    
    if not os.path.exists(local_path):
        return None
    
    print(f"Resuming from local checkpoint: {local_path}")
    return torch.load(local_path, map_location=config.device, weights_only=False)


def save_checkpoint(
    model, optimizer, config, iter_num: int, val_loss: float,
    best_val_loss: float, tag: str = 'latest'
):
    """Save training checkpoint."""
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
    
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'iter_num': iter_num,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    }
    
    path = os.path.join(config.out_dir, 'ckpt.pt')
    torch.save(checkpoint, path)
    print(f"  -> saved checkpoint to {path} ({tag}, val {val_loss:.4f})")


# --- Training Loop ---
def train(model, optimizer, config, print_logs: bool = False):
    """Training loop with learning rate scheduling and checkpointing."""
    losses = []
    start_time = 0
    iter_num = 0
    best_val_loss = float('inf')
    
    # Try to resume
    ckpt = try_resume(config, config.out_dir)
    if ckpt:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        iter_num = ckpt['iter_num'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
    
    while iter_num < config.max_iters:
        # Update learning rate
        lr = get_training_lr(iter_num, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Periodic evaluation
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            losses_dict = estimate_loss(model, config)
            losses.append(losses_dict)
            
            elapsed = 0
            print(f"step {iter_num:>6d} | train {losses_dict['train']:.4f} | val {losses_dict['val']:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            
            # Save checkpoint
            is_best = losses_dict['val'] < best_val_loss
            if is_best:
                best_val_loss = losses_dict['val']
            
            save_checkpoint(model, optimizer, config, iter_num, 
                          losses_dict['val'], best_val_loss,
                          tag='best' if is_best else 'latest')
        
        # Training step
        if hasattr(config, 'train_bin') and os.path.exists(config.train_bin):
            x, y = get_batch('memmap', config, split='train')
        else:
            x, y = get_batch('char', config, split='train')
        
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        
        if iter_num % config.log_interval == 0 and iter_num > 0:
            print(f"  iter {iter_num} | minibatch loss {loss.item():.4f} | lr {lr:.2e}")
        
        iter_num += 1
    
    # Final evaluation
    final_losses = estimate_loss(model, config)
    print(f"\nTraining complete!")
    print(f"Final train loss: {final_losses['train']:.4f}")
    print(f"Final val loss: {final_losses['val']:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, config, iter_num, final_losses['val'], best_val_loss, 'final')
    
    return losses


# --- Text Generation ---
@torch.no_grad()
def generate_text(
    model,
    config,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    num_samples: int = 3
):
    """Generate text from trained model with top-k sampling.
    
    Args:
        model: Trained GPT model
        config: Training config
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (<1 greedy, >1 creative)
        top_k: Only sample from top-k most likely tokens
        num_samples: Number of independent samples
        
    Returns:
        List of generated text strings
    """
    model.eval()
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)

    results = []
    block_size = config.context_window
    device = getattr(config, 'device', 'cpu')

    for sample_idx in range(num_samples):
        print(f"\n--- Sample {sample_idx + 1} ---")
        print(f"Prompt: {prompt}")

        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        # Decode
        generated = tokenizer.decode(idx[0].cpu().tolist())
        print(f"\n{generated}")
        results.append(generated)

    return results


# --- Main Training Entry Point ---
def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Scrappy LLM')
    parser.add_argument('--data', type=str, default='./data/tinyshakespeare.txt',
                       help='Path to training data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--model_type', type=str, default='llama', choices=['llama'],
                       help='Model architecture type')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, mps, cpu')
    parser.add_argument('--dtype', type=str, default='float32',
                       help='Training dtype: float32, float16')
    parser.add_argument('--context_window', type=int, default=128,
                       help='Context window size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', '--max_iters', type=int, default=5000,
                       help='Number of training iterations')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--n_layer', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--n_head', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--out_dir', type=str, default='./out',
                       help='Output directory for checkpoints')
    args = parser.parse_args()
    
    # Build config
    config_dict = {
        'device': args.device,
        'dtype': args.dtype,
        'context_window': args.context_window,
        'batch_size': args.batch_size,
        'max_iters': args.epochs,
        'd_model': args.d_model,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'learning_rate': args.learning_rate,
        'out_dir': args.out_dir,
        'run_name': f'scrappy-{time.strftime("%Y%m%d-%H%M%S")}'
    }
    
    # Load data
    print("\n=== Loading Data ===")
    with open(args.data, 'r') as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters from {args.data}")

    # Initialize tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")

    # Encode text
    encoded = tokenizer.encode(text)
    config_dict['vocab_size'] = tokenizer.vocab_size
    
    print(f"Dataset loaded: {len(encoded)} tokens")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create config
    from config import Config
    config = Config(**config_dict)
    
    # Split data
    train_size = int(0.8 * len(encoded))
    val_size = int(0.1 * len(encoded))

    train_data = torch.tensor(encoded[:train_size], dtype=torch.long)
    val_data = torch.tensor(encoded[train_size:train_size + val_size], dtype=torch.long)

    config.data_dict = {'train': train_data, 'val': val_data}

    # Setup model
    from training.model import GPT, ModelConfig
    model_config = ModelConfig(
        vocab_size=config.vocab_size,
        block_size=config.context_window,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.d_model
    )
    
    print("\n=== Building Model ===")

    # Set device
    if config.device == 'auto':
        if torch.cuda.is_available():
            config.device = 'cuda'
        elif torch.backends.mps.is_available():
            config.device = 'mps'
        else:
            config.device = 'cpu'

    model = GPT(model_config).to(config.device)
    
    if config.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)

    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Train
    os.makedirs(config.out_dir, exist_ok=True)
    if os.path.exists(os.path.join(config.out_dir, 'ckpt.pt')):
        print("Found existing checkpoint, will resume training")
    else:
        print("Starting fresh training")

    train(model, optimizer, config, print_logs=True)
    
    # Generate sample
    print("\n=== Generating Text ===")
    sample_text = "To be or not to be"
    generate_text(model, config, tokenizer, sample_text,
                 max_new_tokens=100, temperature=0.8, top_k=40, num_samples=3)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
