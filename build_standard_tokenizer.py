#!/usr/bin/env python
"""
Build a standard BPE tokenizer with fixed vocab size.
This tokenizer can be used across all training and fine-tuning tasks.
"""

import os
import json
import sys
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder


def collect_data_files(data_paths):
    """Resolve file and directory paths into a flat list of .txt/.csv files."""
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    resolved_files = []
    for path in data_paths:
        p = Path(path)
        if p.is_dir():
            matched = sorted(p.rglob("*.txt")) + sorted(p.rglob("*.csv"))
            if not matched:
                print(f"Warning: no .txt or .csv files found in directory: {path}")
            else:
                resolved_files.extend(str(x) for x in matched)
        elif p.is_file():
            resolved_files.append(str(p))
        else:
            print(f"Warning: path not found, skipping: {path}")

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file_path in resolved_files:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)

    return unique_files


def build_standard_tokenizer(
    data_files: list,
    vocab_size: int = 8192,
    output_path: str = "./out/standard_tokenizer.json"
):
    """Build a BPE tokenizer from multiple data sources.

    Args:
        data_files: List of text/csv files to build vocabulary from
        vocab_size: Target vocabulary size
        output_path: Where to save the tokenizer

    Returns:
        Tokenizer instance
    """
    print(f"Building standard tokenizer with vocab_size={vocab_size}")
    print(f"Input files: {data_files}")

    # Collect all training text into temp files
    temp_files = []
    all_text = []

    for file_path in data_files:
        print(f"\nReading: {file_path}")

        if not os.path.exists(file_path):
            print(f"  Warning: File not found, skipping")
            continue

        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                all_text.append(text)
                print(f"  Added {len(text)} characters")
                temp_files.append(file_path)

        elif file_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(file_path)

            # Combine all text columns
            text_parts = []
            for col in df.columns:
                if col not in ['', 'Unnamed: 0']:  # Skip index columns
                    text_parts.extend(df[col].astype(str).tolist())

            combined = "\n".join(text_parts)
            all_text.append(combined)
            print(f"  Added {len(combined)} characters from {len(df)} rows")

            # Save to temp file for tokenizer training
            temp_csv_txt = file_path + ".tmp.txt"
            with open(temp_csv_txt, 'w', encoding='utf-8') as f:
                f.write(combined)
            temp_files.append(temp_csv_txt)

    # Combine all text
    training_text = "\n".join(all_text)
    print(f"\nTotal training text: {len(training_text)} characters")

    # Build BPE tokenizer using HuggingFace tokenizers
    print(f"Training BPE tokenizer with vocab_size={vocab_size}...")

    tokenizer = HFTokenizer(BPE(unk_token="<UNK>"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        show_progress=True
    )

    tokenizer.pre_tokenizer = Whitespace()

    # IMPORTANT: Add decoder to properly handle spaces
    tokenizer.decoder = BPEDecoder(suffix="</w>")

    # Train on all files
    tokenizer.train(files=temp_files, trainer=trainer)

    # Clean up temp CSV files
    for tf in temp_files:
        if tf.endswith('.tmp.txt'):
            os.remove(tf)

    print(f"Tokenizer trained: {tokenizer.get_vocab_size()} tokens")

    # Save tokenizer
    tokenizer.save(output_path)
    print(f"Saved to: {output_path}")

    # Test tokenization
    test_text = "Q: hi, how are you doing?\nA: i'm fine. how about yourself?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)

    print(f"\nTest encoding:")
    print(f"  Original: {test_text[:80]}")
    print(f"  Tokens: {len(encoded.ids)} tokens")
    print(f"  Decoded: {decoded[:80]}")

    return tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build standard tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8192,
                        help="Vocabulary size")
    parser.add_argument("--output", type=str, default="./out/standard_tokenizer.json",
                        help="Output path")
    parser.add_argument("--data", nargs="+",
                        default=["./data"],
                        help="Data files or directories to train tokenizer on")

    args = parser.parse_args()

    data_files = collect_data_files(args.data)
    if not data_files:
        print("Error: no valid .txt or .csv files found in the provided data paths.")
        sys.exit(1)

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Build tokenizer
    tokenizer = build_standard_tokenizer(
        data_files=data_files,
        vocab_size=args.vocab_size,
        output_path=args.output
    )

    print(f"\n✓ Standard tokenizer ready!")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  Path: {args.output}")
    print(f"\nNow you can:")
    print(f"  1. Train from scratch with this tokenizer")
    print(f"  2. Fine-tune with this tokenizer")
    print(f"  3. Use it consistently across all models")
