#!/usr/bin/env python
"""
Fine-tune Scrappy LLM on conversation data to create a chatbot.
Uses the Conversation.csv dataset with question-answer pairs.
"""

import torch
import pandas as pd
import os
from pathlib import Path
from training import Config, CharTokenizer, GPT, ModelConfig, Tokenizer
from training.sft_trainer import load_pretrained_checkpoint, save_sft_checkpoint, get_lr, estimate_sft_val


def load_conversation_data(csv_path: str, tokenizer, train_split: float = 0.9, max_context: int = 128):
    """Load conversation data from CSV and format for training.

    Args:
        csv_path: Path to Conversation.csv
        tokenizer: Tokenizer instance
        train_split: Fraction of data for training (rest for validation)
        max_context: Maximum context length

    Returns:
        train_examples, val_examples: Lists of tokenized sequences
    """
    print(f"Loading conversation data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Check columns
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("CSV must have 'question' and 'answer' columns")

    train_examples = []
    val_examples = []

    total_rows = len(df)
    train_size = int(total_rows * train_split)

    for idx, row in df.iterrows():
        question = str(row['question']).strip()
        answer = str(row['answer']).strip()

        # Format as conversational turn
        conversation = f"Q: {question}\nA: {answer}"

        # Tokenize
        token_ids = tokenizer.encode(conversation)

        # Truncate if too long
        if len(token_ids) > max_context:
            token_ids = token_ids[:max_context]

        # Split into train/val
        if idx < train_size:
            train_examples.append(token_ids)
        else:
            val_examples.append(token_ids)

    print(f"Loaded {len(train_examples)} training examples, {len(val_examples)} validation examples")
    return train_examples, val_examples


def get_batch(examples, batch_size: int, context_window: int, device: str):
    """Sample a batch from conversation examples.

    Args:
        examples: List of token sequences
        batch_size: Batch size
        context_window: Context window size
        device: Device to use

    Returns:
        x, y: Input and target tensors
    """
    if len(examples) < batch_size:
        raise ValueError(f"Not enough examples: {len(examples)} < {batch_size}")

    # Randomly sample batch_size examples
    ix = torch.randint(len(examples), (batch_size,))

    x_list = []
    y_list = []

    for i in ix:
        seq = examples[i.item()]

        # Pad or truncate to context_window
        if len(seq) < context_window:
            seq = seq + [0] * (context_window - len(seq))
        else:
            seq = seq[:context_window]

        # x is the sequence, y is shifted by 1
        x_list.append(seq)
        y_list.append(seq[1:] + [0])

    x = torch.tensor(x_list, dtype=torch.long, device=device)
    y = torch.tensor(y_list, dtype=torch.long, device=device)

    return x, y


def _run_training_loop(model, tokenizer, train_examples, val_examples, optimizer, config, learning_rate, warmup_iters, max_iters, eval_interval, log_interval, batch_size, context_window, device, out_dir):
    """Run the training loop (extracted to avoid code duplication)."""
    model.train()
    best_val_loss = float('inf')

    print(f"\nStarting training for {max_iters} iterations...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    for iter_num in range(max_iters):
        # Learning rate schedule
        if iter_num < warmup_iters:
            lr = learning_rate * (iter_num + 1) / (warmup_iters + 1)
        else:
            lr = learning_rate

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Validation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            val_loss = estimate_loss(model, val_examples, config, num_batches=10)
            print(f"\nStep {iter_num:6d} | val loss {val_loss:.4f} | lr {lr:.2e}")

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"  -> New best validation loss!")

            raw_model = model.module if hasattr(model, 'module') else model
            raw_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': model.config,
                'iter_num': iter_num,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'stage': 'chatbot_sft'
            }

            ckpt_path = os.path.join(out_dir, 'chatbot_best.pt' if is_best else 'chatbot_latest.pt')
            torch.save(checkpoint, ckpt_path)
            print(f"  -> Saved checkpoint to {ckpt_path}")

        # Training step
        x, y = get_batch(train_examples, batch_size, context_window, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Logging
        if iter_num % log_interval == 0:
            print(f"  iter {iter_num:6d} | train loss {loss.item():.4f} | lr {lr:.2e}")

    print(f"\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {out_dir}/chatbot_best.pt")

    return model, tokenizer


def estimate_loss(model, examples, config, num_batches: int = 10):
    """Estimate validation loss."""
    model.eval()
    losses = []

    context_window = getattr(config, 'context_window', getattr(config, 'block_size', 128))

    for _ in range(min(num_batches, len(examples) // config.batch_size)):
        x, y = get_batch(examples, config.batch_size, context_window, config.device)
        with torch.no_grad():
            _, loss = model(x, y)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def finetune_chatbot(
    pretrained_ckpt: str = None,
    data_path: str = "./data/Conversation.csv",
    out_dir: str = "./out",
    device: str = "cuda",
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    max_iters: int = 5000,
    eval_interval: int = 500,
    log_interval: int = 100,
    warmup_iters: int = 100,
    train_from_scratch: bool = False,
    standard_tokenizer: str = None
):
    """Fine-tune model on conversation data.

    Args:
        pretrained_ckpt: Path to pretrained checkpoint (None to train from scratch)
        data_path: Path to Conversation.csv
        out_dir: Output directory for checkpoints
        device: Device to use
        batch_size: Batch size
        learning_rate: Learning rate
        max_iters: Maximum training iterations
        eval_interval: Steps between validation
        log_interval: Steps between logging
        warmup_iters: Warmup iterations
        train_from_scratch: If True, initialize new model instead of loading checkpoint
        standard_tokenizer: Path to standard tokenizer JSON (recommended for consistency)
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Fine-tuning Chatbot ===")
    print(f"Device: {device}")
    print(f"Data: {data_path}")
    print(f"Output: {out_dir}")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # Load or create tokenizer
    tokenizer_path = os.path.join(out_dir, "tokenizer.json")

    # Priority 1: Use standard tokenizer if provided
    if standard_tokenizer and os.path.exists(standard_tokenizer):
        print(f"Loading standard tokenizer: {standard_tokenizer}")
        tokenizer = Tokenizer.auto_load(standard_tokenizer)
        vocab_size = tokenizer.vocab_size
        print(f"Standard tokenizer loaded: vocab_size={vocab_size}")

        # If training from scratch, use this tokenizer
        if train_from_scratch or not pretrained_ckpt:
            context_window = 128
            model_config = ModelConfig(
                vocab_size=vocab_size,
                block_size=context_window,
                n_layer=4,
                n_head=8,
                n_embd=256
            )
            model = GPT(model_config).to(device)
            print(f"Created model: {model.num_parameters() / 1e6:.2f}M parameters")
            if device == 'cuda' and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)


            # Load conversation data and start training
            train_examples, val_examples = load_conversation_data(data_path, tokenizer, max_context=context_window)

            if len(train_examples) == 0:
                raise ValueError("No training examples loaded!")

            # Skip the rest of tokenizer/model loading
            config = Config(
                vocab_size=model.config.vocab_size,
                batch_size=batch_size,
                context_window=context_window,
                device=device,
                out_dir=out_dir,
                max_iters=max_iters,
                eval_interval=eval_interval,
                log_interval=log_interval,
                warmup_iters=warmup_iters,
                sft_lr=learning_rate,
                min_lr=learning_rate * 0.1,
                lr_decay_iters=max_iters
            )

            # Setup optimizer
            decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
            no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

            optim_groups = [
                {'params': decay_params, 'weight_decay': config.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]

            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

            # Jump to training loop
            return _run_training_loop(model, tokenizer, train_examples, val_examples, optimizer, config, learning_rate, warmup_iters, max_iters, eval_interval, log_interval, batch_size, context_window, device, out_dir)

        # If using pretrained model, load it but verify vocab size matches
        elif pretrained_ckpt and os.path.exists(pretrained_ckpt):
            print(f"Loading pretrained checkpoint: {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location=device, weights_only=False)

            ckpt_config = checkpoint['config']
            if isinstance(ckpt_config, dict):
                ckpt_vocab_size = ckpt_config.get('vocab_size', 8192)
            else:
                ckpt_vocab_size = getattr(ckpt_config, 'vocab_size', 8192)

            if ckpt_vocab_size != vocab_size:
                print(f"\n⚠ WARNING: Vocab size mismatch!")
                print(f"  Checkpoint: {ckpt_vocab_size}")
                print(f"  Standard tokenizer: {vocab_size}")
                print(f"  Cannot load pretrained weights - training from scratch instead.")
                train_from_scratch = True
            else:
                # Load model with matching vocab
                # First, infer actual dimensions from state dict
                model_state = checkpoint['model']
                actual_n_embd = None
                actual_n_layer = None
                actual_block_size = None

                if 'wte.weight' in model_state:
                    actual_n_embd = model_state['wte.weight'].shape[1]
                if 'wpe.weight' in model_state:
                    actual_block_size = model_state['wpe.weight'].shape[0]

                # Count layers
                layer_indices = set()
                for key in model_state.keys():
                    if key.startswith('layers.'):
                        parts = key.split('.')
                        if len(parts) > 1 and parts[1].isdigit():
                            layer_indices.add(int(parts[1]))
                if layer_indices:
                    actual_n_layer = max(layer_indices) + 1

                # Extract from config with fallbacks to inferred values
                if isinstance(ckpt_config, dict):
                    context_window = actual_block_size or ckpt_config.get('context_window', ckpt_config.get('block_size', 128))
                    n_layer = actual_n_layer or ckpt_config.get('n_layer', 4)
                    n_head = ckpt_config.get('n_head', 8)
                    n_embd = actual_n_embd or ckpt_config.get('d_model', ckpt_config.get('n_embd', 128))
                else:
                    context_window = actual_block_size or getattr(ckpt_config, 'context_window', getattr(ckpt_config, 'block_size', 128))
                    n_layer = actual_n_layer or getattr(ckpt_config, 'n_layer', 4)
                    n_head = getattr(ckpt_config, 'n_head', 8)
                    n_embd = actual_n_embd or getattr(ckpt_config, 'd_model', getattr(ckpt_config, 'n_embd', 128))

                print(f"Model architecture: {n_layer} layers, {n_embd} dims, {context_window} context")

                model_config = ModelConfig(
                    vocab_size=vocab_size,
                    block_size=context_window,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd
                )
                model = GPT(model_config).to(device)
                model.load_state_dict(checkpoint['model'])
                print(f"Loaded model: {model.num_parameters() / 1e6:.2f}M parameters")
                if device == 'cuda' and torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)


                # Continue to training with loaded model
                train_examples, val_examples = load_conversation_data(data_path, tokenizer, max_context=context_window)

                if len(train_examples) == 0:
                    raise ValueError("No training examples loaded!")

                config = Config(
                    vocab_size=model.config.vocab_size,
                    batch_size=batch_size,
                    context_window=context_window,
                    device=device,
                    out_dir=out_dir,
                    max_iters=max_iters,
                    eval_interval=eval_interval,
                    log_interval=log_interval,
                    warmup_iters=warmup_iters,
                    sft_lr=learning_rate,
                    min_lr=learning_rate * 0.1,
                    lr_decay_iters=max_iters
                )

                decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
                no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

                optim_groups = [
                    {'params': decay_params, 'weight_decay': config.weight_decay},
                    {'params': no_decay_params, 'weight_decay': 0.0}
                ]

                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

                return _run_training_loop(model, tokenizer, train_examples, val_examples, optimizer, config, learning_rate, warmup_iters, max_iters, eval_interval, log_interval, batch_size, context_window, device, out_dir)

    # Fall through to original tokenizer loading logic if no standard tokenizer

    if pretrained_ckpt and os.path.exists(pretrained_ckpt) and not train_from_scratch:
        # Load tokenizer from checkpoint
        print(f"Loading pretrained checkpoint: {pretrained_ckpt}")
        checkpoint = torch.load(pretrained_ckpt, map_location=device, weights_only=False)

        # Extract config and infer actual dimensions from state dict
        ckpt_config = checkpoint['config']
        model_state = checkpoint['model']

        # Infer actual dimensions
        actual_n_embd = None
        actual_n_layer = None
        actual_block_size = None

        if 'wte.weight' in model_state:
            actual_n_embd = model_state['wte.weight'].shape[1]
        if 'wpe.weight' in model_state:
            actual_block_size = model_state['wpe.weight'].shape[0]

        # Count layers
        layer_indices = set()
        for key in model_state.keys():
            if key.startswith('layers.'):
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    layer_indices.add(int(parts[1]))
        if layer_indices:
            actual_n_layer = max(layer_indices) + 1

        # Extract from config with fallbacks
        if isinstance(ckpt_config, dict):
            vocab_size = ckpt_config.get('vocab_size', 8192)
            context_window = actual_block_size or ckpt_config.get('context_window', ckpt_config.get('block_size', 128))
            n_layer = actual_n_layer or ckpt_config.get('n_layer', 4)
            n_head = ckpt_config.get('n_head', 8)
            n_embd = actual_n_embd or ckpt_config.get('d_model', ckpt_config.get('n_embd', 128))
        else:
            vocab_size = getattr(ckpt_config, 'vocab_size', 8192)
            context_window = actual_block_size or getattr(ckpt_config, 'context_window', getattr(ckpt_config, 'block_size', 128))
            n_layer = actual_n_layer or getattr(ckpt_config, 'n_layer', 4)
            n_head = getattr(ckpt_config, 'n_head', 8)
            n_embd = actual_n_embd or getattr(ckpt_config, 'd_model', getattr(ckpt_config, 'n_embd', 128))

        print(f"Model architecture: {n_layer} layers, {n_embd} dims, {context_window} context")

        # Try to load or recreate the ORIGINAL tokenizer from checkpoint
        if 'tokenizer' in checkpoint and checkpoint['tokenizer'] is not None:
            print("Loading tokenizer from checkpoint...")
            tokenizer = checkpoint['tokenizer']
        elif os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}")
            # Load the saved CharTokenizer
            import json
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tok_data = json.load(f)

            if tok_data.get('type') == 'CharTokenizer':
                # Reconstruct CharTokenizer from saved data
                tokenizer = CharTokenizer.__new__(CharTokenizer)
                tokenizer.vocab_size = tok_data['vocab_size']
                tokenizer.stoi = tok_data['stoi']
                tokenizer.itos = {int(k): v for k, v in tok_data['itos'].items()}
                print(f"Loaded CharTokenizer with vocab_size={tokenizer.vocab_size}")
            else:
                # Try regular Tokenizer class
                tokenizer = Tokenizer(tokenizer_path)
        else:
            # Need to recreate tokenizer from ORIGINAL training data
            # Check if original data path is in checkpoint
            original_data = None
            if isinstance(ckpt_config, dict):
                original_data = ckpt_config.get('data_path')
            else:
                original_data = getattr(ckpt_config, 'data_path', None)

            if original_data and os.path.exists(original_data):
                print(f"Recreating tokenizer from original training data: {original_data}")
                with open(original_data, 'r', encoding='utf-8') as f:
                    original_text = f.read()
                tokenizer = CharTokenizer(original_text)
            else:
                # Fallback: combine original vocab with new data
                print("Warning: Cannot find original tokenizer or training data.")
                print(f"Model expects vocab_size={vocab_size}, but this may not match.")
                print("Consider training from scratch with --from-scratch flag.")

                # Try to infer the original tokenizer by creating one with the right vocab size
                df = pd.read_csv(data_path)
                all_text = " ".join(df['question'].astype(str) + " " + df['answer'].astype(str))
                tokenizer = CharTokenizer(all_text)

                if tokenizer.vocab_size != vocab_size:
                    raise ValueError(
                        f"Tokenizer vocab mismatch: checkpoint has {vocab_size}, "
                        f"but conversation data has {tokenizer.vocab_size} unique chars.\n"
                        f"Solutions:\n"
                        f"  1. Train from scratch: add --from-scratch flag\n"
                        f"  2. Provide original training data to recreate tokenizer\n"
                        f"  3. Save tokenizer.json during pretraining for reuse"
                    )

        # Create model and load weights
        model_config = ModelConfig(
            vocab_size=vocab_size,
            block_size=context_window,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd
        )
        model = GPT(model_config).to(device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model: {model.num_parameters() / 1e6:.2f}M parameters")
        if device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    else:
        # Train from scratch - create tokenizer and model
        print("Training from scratch...")
        df = pd.read_csv(data_path)
        all_text = " ".join(df['question'].astype(str) + " " + df['answer'].astype(str))

        print("Creating character tokenizer...")
        tokenizer = CharTokenizer(all_text)

        # Save tokenizer
        if hasattr(tokenizer, 'save'):
            tokenizer.save(tokenizer_path)
            print(f"Saved tokenizer to {tokenizer_path}")

        # Create model
        context_window = 128
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=context_window,
            n_layer=4,
            n_head=8,
            n_embd=256
        )
        model = GPT(model_config).to(device)
        print(f"Created model: {model.num_parameters() / 1e6:.2f}M parameters")
        if device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)


    # Load conversation data
    context_window = model.config.block_size
    train_examples, val_examples = load_conversation_data(data_path, tokenizer, max_context=context_window)

    if len(train_examples) == 0:
        raise ValueError("No training examples loaded!")

    # Create config
    config = Config(
        vocab_size=model.config.vocab_size,
        batch_size=batch_size,
        context_window=context_window,
        device=device,
        out_dir=out_dir,
        max_iters=max_iters,
        eval_interval=eval_interval,
        log_interval=log_interval,
        warmup_iters=warmup_iters,
        sft_lr=learning_rate,
        min_lr=learning_rate * 0.1,
        lr_decay_iters=max_iters
    )

    # Setup optimizer
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

    # Training loop
    model.train()
    best_val_loss = float('inf')

    print(f"\nStarting training for {max_iters} iterations...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    for iter_num in range(max_iters):
        # Learning rate schedule
        if iter_num < warmup_iters:
            lr = learning_rate * (iter_num + 1) / (warmup_iters + 1)
        else:
            lr = learning_rate

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Validation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            val_loss = estimate_loss(model, val_examples, config, num_batches=10)
            print(f"\nStep {iter_num:6d} | val loss {val_loss:.4f} | lr {lr:.2e}")

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"  -> New best validation loss!")

            raw_model = model.module if hasattr(model, 'module') else model
            raw_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': model.config,
                'iter_num': iter_num,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'stage': 'chatbot_sft'
            }

            ckpt_path = os.path.join(out_dir, 'chatbot_best.pt' if is_best else 'chatbot_latest.pt')
            torch.save(checkpoint, ckpt_path)
            print(f"  -> Saved checkpoint to {ckpt_path}")

        # Training step
        x, y = get_batch(train_examples, batch_size, context_window, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Logging
        if iter_num % log_interval == 0:
            print(f"  iter {iter_num:6d} | train loss {loss.item():.4f} | lr {lr:.2e}")

    print(f"\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {out_dir}/chatbot_best.pt")

    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune chatbot on conversation data")
    parser.add_argument("--data", type=str, default="./data/Conversation.csv", help="Path to conversation CSV")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained checkpoint (optional)")
    parser.add_argument("--out", type=str, default="./out", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/mps/auto)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--iters", type=int, default=5000, help="Max iterations")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to standard tokenizer (recommended)")

    args = parser.parse_args()

    model, tokenizer = finetune_chatbot(
        pretrained_ckpt=args.pretrained,
        data_path=args.data,
        out_dir=args.out,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_iters=args.iters,
        eval_interval=args.eval_interval,
        train_from_scratch=args.from_scratch,
        standard_tokenizer=args.tokenizer
    )

    print("\n=== Testing Chatbot ===")
    from query import query_model

    test_questions = [
        "hi, how are you doing?",
        "what school do you go to?",
        "how's it going?"
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        response = query_model(model, tokenizer, f"Q: {question}\nA:", max_tokens=50, temperature=0.7)
        print(f"A: {response}")
