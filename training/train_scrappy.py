"""
Main training script for Scrappy LLM.
Integrates all improvements from pirate_llm while preserving your model design.
"""

import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Local imports
from .config import Config
from .tokenizer import Tokenizer, CharTokenizer
from .data_loader import get_batch
from .data_prep import prepare_dataset, load_tokenized_data
from .sft_trainer import sft_train, get_sft_batch
from .model import GPT, RMSNorm, RoPEAttention, SwiGLU, LlamaModel, ModelConfig

# Import training utilities
import math
from contextlib import nullcontext


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Scrappy LLM with pirate_llm improvements"
    )
    
    # Core training args
    parser.add_argument(
        '--data', type=str, default='./data/tinyshakespeare.txt',
        help='Path to training data (text file)'
    )
    parser.add_argument(
        '--out_dir', type=str, default='./out',
        help='Output directory for checkpoints'
    )
    
    # Model config
    parser.add_argument(
        '--context_window', type=int, default=128,
        help='Context window size (tokens/characters)'
    )
    parser.add_argument(
        '--d_model', type=int, default=128,
        help='Model hidden dimension'
    )
    parser.add_argument(
        '--n_layer', type=int, default=4,
        help='Number of transformer layers'
    )
    parser.add_argument(
        '--n_head', type=int, default=8,
        help='Number of attention heads'
    )
    
    # Training args
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs', '--max_iters', type=int, default=5000,
        help='Number of training iterations'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device: auto, cuda, mps, cpu'
    )
    parser.add_argument(
        '--dtype', type=str, default='float32',
        help='Training dtype'
    )
    
    # Advanced training args (from pirate_llm)
    parser.add_argument(
        '--weight_decay', type=float, default=0.1,
        help='Weight decay for L2 regularization'
    )
    parser.add_argument(
        '--beta1', type=float, default=0.9,
        help='Adam beta1 parameter'
    )
    parser.add_argument(
        '--beta2', type=float, default=0.95,
        help='Adam beta2 parameter'
    )
    parser.add_argument(
        '--grad_clip', type=float, default=1.0,
        help='Gradient clipping threshold'
    )
    parser.add_argument(
        '--warmup_iters', type=int, default=200,
        help='Learning rate warmup iterations'
    )
    parser.add_argument(
        '--lr_decay_iters', type=int, default=5000,
        help='Learning rate decay iterations'
    )
    parser.add_argument(
        '--min_lr', type=float, default=3e-5,
        help='Minimum learning rate'
    )
    
    # SFT fine-tuning
    parser.add_argument(
        '--sft', action='store_true',
        help='Enable SFT fine-tuning mode'
    )
    parser.add_argument(
        '--sft_dataset', type=str, default=None,
        help='Path to SFT dataset (instruction-response CSV)'
    )
    parser.add_argument(
        '--sft_lr', type=float, default=2e-5,
        help='Learning rate for SFT'
    )
    parser.add_argument(
        '--pretrained', type=str, default=None,
        help='Path to pre-trained checkpoint for SFT'
    )
    
    # Generation
    parser.add_argument(
        '--prompt', type=str, default='Once upon a time',
        help='Prompt for text generation'
    )
    parser.add_argument(
        '--generate', action='store_true',
        help='Generate text after training'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=100,
        help='Max tokens to generate'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='Generation temperature'
    )
    parser.add_argument(
        '--top_k', type=int, default=40,
        help='Top-k sampling (0 = disabled)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=3,
        help='Number of generated samples'
    )
    
    # Checkpoint handling
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from existing checkpoint'
    )
    parser.add_argument(
        '--clear_checkpoint', action='store_true',
        help='Delete existing checkpoint before training (fresh start)'
    )
    parser.add_argument(
        '--eval_interval', type=int, default=100,
        help='Evaluate every N iterations'
    )
    parser.add_argument(
        '--log_interval', type=int, default=10,
        help='Log every N iterations'
    )
    parser.add_argument(
        '--eval_iters', type=int, default=10,
        help='Number of batches for eval'
    )
    
    # Run name
    parser.add_argument(
        '--run_name', type=str, default=None,
        help='Custom run name'
    )

    # Tokenizer
    parser.add_argument(
        '--tokenizer', type=str, default=None,
        help='Path to pre-trained tokenizer JSON (BPE or char). Creates CharTokenizer from data if not provided.'
    )

    return parser.parse_args()


def setup_config(args, model_config=None):
    """Build configuration from command-line args."""
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    run_name = args.run_name or f'scrappy-{time.strftime("%Y%m%d-%H%M%S")}'
    
    return Config(
        run_name=run_name,
        out_dir=args.out_dir,
        device=device,
        dtype=args.dtype,
        context_window=args.context_window,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        batch_size=args.batch_size,
        max_iters=args.epochs,
        learning_rate=args.learning_rate,
        
        # Pirate_llm improvements
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        
        # LR schedule
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        min_lr=args.min_lr,
        
        # Training loop
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        eval_iters=args.eval_iters,
    )


def train_scrappy(model, optimizer, config, print_logs=False):
    """
    Training loop with pirate_llm-style improvements.
    """
    losses = []
    iter_num = 0
    best_val_loss = float('inf')
    start_time = 0
    
    # Try to resume
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    if config.resume or os.path.exists(ckpt_path):
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            try:
                ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                iter_num = ckpt['iter_num'] + 1
                best_val_loss = ckpt.get('best_val_loss', float('inf'))
                print(f"[OK] Successfully resumed from iteration {iter_num}")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"\n[WARNING] Checkpoint architecture mismatch!")
                    print(f"   Checkpoint was saved with different model parameters.")
                    print(f"   Error: {str(e)[:200]}...")
                    print(f"\n   Solutions:")
                    print(f"   1. Use --clear_checkpoint flag to delete and start fresh")
                    print(f"   2. Use --out_dir to specify a different directory")
                    print(f"   3. Delete manually: rm {ckpt_path}")
                    print(f"\n   Starting fresh training instead...\n")
                    # Reset to fresh training
                    iter_num = 0
                    best_val_loss = float('inf')
                else:
                    # Re-raise if it's a different error
                    raise
    
    print(f"Starting training at iteration {iter_num}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Context window: {config.context_window}")
    
    while iter_num < config.max_iters:
        # Update learning rate
        lr = get_lr(iter_num, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Periodic evaluation
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            losses_dict = evaluate_model(model, config)
            losses.append(losses_dict)
            
            elapsed = 0
            print(f"step {iter_num:>7d} | train {losses_dict['train']:.4f} | val {losses_dict['val']:.4f} | lr {lr:.2e}")
            
            # Save checkpoint
            is_best = losses_dict['val'] < best_val_loss
            if is_best:
                best_val_loss = losses_dict['val']
            
            save_checkpoint(model, optimizer, config, iter_num, 
                          losses_dict['val'], best_val_loss,
                          tag='best' if is_best else 'latest')
            
            start_time = iter_num
        
        # Training step
        if hasattr(config, 'train_bin') and os.path.exists(config.train_bin):
            x, y = get_batch('memmap', config, 'train')
        else:
            x, y = get_batch('char', config, 'train')
        
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        
        # Logging
        if iter_num % config.log_interval == 0 and iter_num > 0:
            print(f"  iter {iter_num} | loss {loss.item():.4f} | lr {lr:.2e}")
        
        iter_num += 1
    
    # Final evaluation
    final_losses = evaluate_model(model, config)
    print(f"\nTraining complete!")
    print(f"Final train loss: {final_losses['train']:.4f}")
    print(f"Final val loss: {final_losses['val']:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, config, iter_num, final_losses['val'], best_val_loss, 'final')
    
    return losses


def get_lr(it, config):
    """Learning rate with warmup + cosine decay."""
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def evaluate_model(model, config):
    """Evaluate model on train and validation sets."""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        
        # Determine data source
        if split == 'train':
            data_file = os.path.join(config.out_dir, 'train.bin')
        else:
            data_file = os.path.join(config.out_dir, 'val.bin')
        
        # For character-level data, use tensor approach
        if hasattr(config, 'data_dict'):
            data = config.data_dict[split]
        else:
            data = load_tokenized_data(data_file)
        
        for k in range(config.eval_iters):
            ix = list(np.random.choice(len(data) - config.context_window - 1, config.batch_size))
            x_batch = torch.stack([data[i:i+config.context_window] for i in ix])
            y_batch = torch.stack([data[i+1:i+config.context_window+1] for i in ix])
            x_batch, y_batch = x_batch.long().to(config.device), y_batch.long().to(config.device)
            
            _, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        
        out[split] = losses.mean().item()
    
    model.train()
    return out


def save_checkpoint(model, optimizer, config, iter_num, val_loss, best_val_loss, tag='latest'):
    """Save training checkpoint."""
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config.__dict__,
        'tokenizer_path': getattr(config, 'tokenizer_path', None),
        'iter_num': iter_num,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    }
    
    path = os.path.join(config.out_dir, 'ckpt.pt')
    torch.save(checkpoint, path)
    print(f"  -> saved checkpoint to {path} ({tag}, val {val_loss:.4f})")


def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40, num_samples=3, device='cpu'):
    """Generate text from trained model."""
    print("\n=== Generating Text ===")

    import torch.nn.functional as F

    model.eval()
    results = []

    for sample_idx in range(num_samples):
        print(f"\n--- Sample {sample_idx + 1} ---")
        print(f"Prompt: {prompt}")

        # Encode prompt
        prompt_ids = tokenizer.encode(prompt)
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        block_size = model.config.block_size if hasattr(model, 'config') else 128

        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_idx), dim=1)

        decoded = tokenizer.decode(idx[0].cpu().tolist())
        print(f"\n{decoded}")
        results.append(decoded)

    print("\n" + "="*60)
    print("\nGeneration complete!")
    return results


def main():
    """Main training and querying script."""
    args = parse_args()
    config = setup_config(args)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Clear checkpoint if requested
    if args.clear_checkpoint:
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"[OK] Cleared checkpoint: {ckpt_path}\n")
    
    # Prepare dataset
    print("\n=== Preparing Dataset ===")
    
    # Load training data
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded {len(text):,} characters from {args.data}")

    # Load or create tokenizer
    if args.tokenizer and os.path.exists(args.tokenizer):
        print(f"Loading tokenizer from {args.tokenizer}")
        from training.tokenizer import Tokenizer
        tokenizer = Tokenizer.auto_load(args.tokenizer)
        vocab_size = tokenizer.vocab_size
        print(f"Loaded tokenizer: vocab_size={vocab_size}")
    else:
        print("Creating CharTokenizer from training data...")
        tokenizer = CharTokenizer(text)
        vocab_size = tokenizer.vocab_size
        print(f"Character vocabulary: {vocab_size} unique characters")

    # Encode text using tokenizer
    print("Tokenizing text...")
    if hasattr(tokenizer, 'hf_tokenizer'):
        # BPE tokenizer - encode in chunks for large files
        chunk_size = 10_000_000  # 10MB chunks
        encoded = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            encoded.extend(tokenizer.encode(chunk))
            progress = min(100, int((i + chunk_size) / len(text) * 100))
            print(f"  Tokenizing: {progress}%", end='\r')
        print(f"  Tokenizing: 100% - Complete!")
    else:
        # CharTokenizer - fast enough for full text
        encoded = tokenizer.encode(text)
    
    # Split data
    train_size = int(0.8 * len(encoded))
    val_size = int(0.1 * len(encoded))
    
    train_data = encoded[:train_size]
    val_data = encoded[train_size:train_size + val_size]
    
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    
    # Save to binary files for streaming
    train_data_arr = np.array(train_data, dtype=np.uint16)
    val_data_arr = np.array(val_data, dtype=np.uint16)
    
    train_path = os.path.join(args.out_dir, 'train.bin')
    val_path = os.path.join(args.out_dir, 'val.bin')
    
    train_data_arr.tofile(train_path)
    val_data_arr.tofile(val_path)
    print(f"Saved training data ({len(train_data_arr):,} tokens) to {train_path}")
    print(f"Saved validation data ({len(val_data_arr):,} tokens) to {val_path}")
    
    # Update config with actual vocab size
    config.vocab_size = vocab_size
    config.tokenizer_path = args.tokenizer  # Save tokenizer path
    config.train_bin = train_path
    config.val_bin = val_path
    config.data_dict = {
        'train': torch.tensor(train_data, dtype=torch.long),
        'val': torch.tensor(val_data, dtype=torch.long)
    }
    
    # Build or load model
    print("\n=== Building Model ===")

    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained model from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=config.device, weights_only=False)

        # Verify tokenizer compatibility
        ckpt_config = ckpt['config']
        ckpt_vocab = ckpt_config.vocab_size if hasattr(ckpt_config, 'vocab_size') else ckpt_config.get('vocab_size')
        if ckpt_vocab != vocab_size:
            raise ValueError(f"Tokenizer mismatch: pretrained has {ckpt_vocab}, current has {vocab_size}")

        # Extract config from checkpoint
        model_config = ModelConfig(
            vocab_size=vocab_size,
            block_size=ckpt_config.context_window if hasattr(ckpt_config, 'context_window') else ckpt_config.get('context_window', ckpt_config.get('block_size', 128)),
            n_layer=ckpt_config.n_layer if hasattr(ckpt_config, 'n_layer') else ckpt_config.get('n_layer'),
            n_head=ckpt_config.n_head if hasattr(ckpt_config, 'n_head') else ckpt_config.get('n_head'),
            n_embd=ckpt_config.d_model if hasattr(ckpt_config, 'd_model') else ckpt_config.get('n_embd', ckpt_config.get('d_model'))
        )
        model = GPT(model_config).to(config.device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded pretrained model: {model.num_parameters() / 1e6:.2f}M parameters")
    else:
        # Create new model
        model_config = ModelConfig(
            vocab_size=vocab_size,
            block_size=config.context_window,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.d_model
        )
        model = GPT(model_config).to(config.device)
        print(f"Created new model: {model.num_parameters() / 1e6:.2f}M parameters")
    
    # Build optimizer
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate,
                                  betas=(config.beta1, config.beta2))
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    config.resume = args.resume
    losses = train_scrappy(model, optimizer, config, print_logs=True)
    
    # Generate sample if requested
    if args.generate:
        generate_response(
            model, tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            num_samples=args.num_samples,
            device=config.device
        )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    main()
