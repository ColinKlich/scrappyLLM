"""
SFT (Supervised Fine-Tuning) utilities for Scrappy LLM.
Adapted from pirate_llm-main training/sft_train.py
"""

import torch
import os
import math


def load_pretrained_checkpoint(ckpt_path: str, config):
    """Load a saved pretraining checkpoint for SFT.
    
    Args:
        ckpt_path: Path to checkpoint file
        config: Updated config for SFT training
        
    Returns:
        GPT model loaded with pretrained weights
    """
    print(f"Loading checkpoint from {ckpt_path}")
    from training.model import GPT, ModelConfig

    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    arch_config = checkpoint['config']

    # Ensure architecture compatibility
    context_window = getattr(config, 'context_window', getattr(config, 'block_size', 128))
    arch_context = getattr(arch_config, 'context_window', getattr(arch_config, 'block_size', 128))

    if context_window != arch_context:
        raise ValueError(
            f"SFT context_window ({context_window}) must equal pretraining "
            f"context_window ({arch_context})"
        )

    # Build ModelConfig from checkpoint config
    if isinstance(arch_config, dict):
        model_config = ModelConfig(
            vocab_size=arch_config.get('vocab_size', 8192),
            block_size=arch_config.get('context_window', arch_config.get('block_size', 128)),
            n_layer=arch_config.get('n_layer', 4),
            n_head=arch_config.get('n_head', 8),
            n_embd=arch_config.get('d_model', arch_config.get('n_embd', 128))
        )
    else:
        model_config = ModelConfig(
            vocab_size=getattr(arch_config, 'vocab_size', 8192),
            block_size=getattr(arch_config, 'context_window', getattr(arch_config, 'block_size', 128)),
            n_layer=getattr(arch_config, 'n_layer', 4),
            n_head=getattr(arch_config, 'n_head', 8),
            n_embd=getattr(arch_config, 'd_model', getattr(arch_config, 'n_embd', 128))
        )

    model = GPT(model_config).to(config.device)
    model.load_state_dict(checkpoint['model'])
    
    print(f"Loaded pretrained model: {model.num_parameters() / 1e6:.2f}M params, "
          f"trained for {checkpoint['iter_num']} iters, val loss {checkpoint['val_loss']:.4f}")
    
    return model


def build_sft_dataset(dataset_path: str, tokenizer):
    """Build SFT dataset from instruction-response pairs.
    
    Args:
        dataset_path: Path to dataset file
        tokenizer: Tokenizer instance
        
    Returns:
        tuple: (train_examples, val_examples) as lists of tokenized sequences
    """
    import pandas as pd
    
    df = pd.read_csv(dataset_path)
    
    train_exs = []
    val_exs = []
    
    for _, row in df.iterrows():
        prompt = str(row['instruction']) if 'instruction' in df.columns else row[0]
        response = str(row['output']) if 'output' in df.columns else row[1]
        
        instruction_text = f"Instruction: {prompt}\nResponse: {response}"
        token_ids = tokenizer.encode(instruction_text)

        context_window = getattr(config, 'context_window', getattr(config, 'block_size', 128))
        if len(token_ids) > context_window:
            token_ids = token_ids[:context_window]
        
        if len(train_exs) < 8:
            val_exs.append(token_ids)
        else:
            train_exs.append(token_ids)
    
    return train_exs, val_exs


def get_sft_batch(examples, config):
    """Sample a batch from SFT examples.
    
    Args:
        examples: List of token ID sequences
        config: Training configuration
        
    Returns:
        x: input tokens
        y: target tokens (shifted)
    """
    batch_size = config.batch_size
    
    # Ensure we have enough examples
    if len(examples) < batch_size:
        raise ValueError(f"Not enough examples: {len(examples)} < {batch_size}")
    
    ix = torch.randint(len(examples), (batch_size,))
    
    x_list = []
    y_list = []
    
    context_window = getattr(config, 'context_window', getattr(config, 'block_size', 128))

    for i in ix:
        seq = examples[i]
        if len(seq) <= context_window:
            # Pad if too short
            seq = seq + [0] * (context_window - len(seq))
        else:
            seq = seq[:context_window]
        
        x_list.append(seq)
        y_list.append(seq[1:] + [0])  # Shifted target
    
    x = torch.tensor(x_list, dtype=torch.long, device=config.device)
    y = torch.tensor(y_list, dtype=torch.long, device=config.device)
    
    return x, y


def save_sft_checkpoint(model, optimizer, config, iter_num: int, val_loss: float, best_val_loss: float, tag: str = "latest"):
    """Save SFT checkpoint.
    
    Args:
        model: GPT model
        optimizer: Optimizer
        config: Config
        iter_num: Current iteration
        val_loss: Validation loss
        best_val_loss: Best validation loss so far
        tag: Checkpoint tag (latest/best)
    """
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'iter_num': iter_num,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'stage': 'sft'
    }
    
    path = os.path.join(config.out_dir, 'sft_ckpt.pt')
    torch.save(checkpoint, path)
    print(f"  -> saved SFT checkpoint to {path} ({tag}, val {val_loss:.4f})")


def sft_train(config, pretrained_ckpt_path: str, sft_dataset_path: str, max_iters: int = None):
    """Train SFT on instruction-response pairs.
    
    Args:
        config: SFT configuration
        pretrained_ckpt_path: Path to pretraining checkpoint
        sft_dataset_path: Path to SFT dataset
        max_iters: Override max iterations
    """
    if max_iters:
        config.max_iters = max_iters
    
    print(f"\n=== SFT run: {config.run_name} ===")
    print(f"Device: {config.device}, dtype: {config.dtype}")
    
    # Load tokenizer if available
    from tokenizer import Tokenizer
    try:
        tokenizer = Tokenizer.from_file(config.tokenizer_path)
    except:
        tokenizer = None
        config.vocab_size = config.context_window + 500  # Estimate for char-level
    
    # Build SFT dataset
    train_examples, val_examples = build_sft_dataset(sft_dataset_path, tokenizer)
    print(f"Loaded {len(train_examples)} training examples, {len(val_examples)} validation examples")
    
    # Load pretrained model
    model = load_pretrained_checkpoint(pretrained_ckpt_path, config)
    
    # Build optimizer with lower learning rate for SFT
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optim_groups, lr=config.sft_lr, betas=(config.beta1, config.beta2))
    
    iter_num = 0
    best_val_loss = float('inf')
    start_time = 0
    
    while iter_num < config.max_iters:
        lr = get_lr(iter_num, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        if iter_num % config.eval_interval == 0:
            val_loss = estimate_sft_val(model, val_examples, config)
            elapsed = 0 if start_time == 0 else (iter_num * 0.01)  # Approximate
            start_time = iter_num
            
            print(f"step {iter_num:>6d} | val {val_loss:.4f} | lr {lr:.2e}")
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            save_sft_checkpoint(model, optimizer, config, iter_num, val_loss, best_val_loss,
                               tag='best' if is_best else 'latest')
        
        x, y = get_sft_batch(train_examples, config)
        
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        
        if iter_num % config.log_interval == 0 and iter_num > 0:
            print(f"  iter {iter_num} | loss {loss.item():.4f} | lr {lr:.2e}")
        
        iter_num += 1
    
    print(f"\nSFT training complete! Best val: {best_val_loss:.4f}")
    return model, optimizer


def get_lr(it: int, config) -> float:
    """Learning rate with warmup + cosine decay."""
    if it < config.warmup_iters:
        return config.sft_lr * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.sft_lr - config.min_lr)


def estimate_sft_val(model, examples, config):
    """Estimate validation loss on SFT data."""
    model.eval()
    losses = []
    
    eval_iters = min(config.eval_iters, len(examples) // config.batch_size)
    
    for _ in range(eval_iters):
        x, y = get_sft_batch(examples, config)
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    
    model.train()
    return torch.tensor(losses).mean().item() if losses else float('inf')
