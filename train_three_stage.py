#!/usr/bin/env python
"""
Three-stage training pipeline for Scrappy LLM:
1. Base model on TinyStories (2.1GB)
2. Fine-tune on Shakespeare (1.2MB)
3. Fine-tune on Conversations (259KB)

All stages use the same standard BPE tokenizer (8192 vocab).
"""

import subprocess
import sys
import os

PYTHON = "C:\\Program Files\\Python311\\python.exe"
TOKENIZER = "./out/standard_tokenizer.json"


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nX {description} FAILED")
        sys.exit(1)
    print(f"\nCheckmark {description} COMPLETED")


def stage1_base_training():
    """Stage 1: Base model on TinyStories (4-8 hours)"""
    cmd = f'''"{PYTHON}" -m training.train_scrappy \
        --data ./data/TinyStoriesV2-GPT4-train.txt \
        --tokenizer {TOKENIZER} \
        --out_dir ./out/stage1 \
        --epochs 50000 \
        --learning_rate 3e-4 \
        --batch_size 32 \
        --context_window 256 \
        --d_model 256 \
        --n_layer 6 \
        --n_head 8 \
        --warmup_iters 2000 \
        --eval_interval 1000 \
        --device cuda \
        --clear_checkpoint
    '''
    run_command(cmd, "STAGE 1: Base Training on TinyStories")


def stage2_shakespeare_finetuning():
    """Stage 2: Fine-tune on Shakespeare (30-60 min)"""
    cmd = f'''"{PYTHON}" -m training.train_scrappy \
        --data ./data/tinyshakespeare.txt \
        --tokenizer {TOKENIZER} \
        --pretrained ./out/stage1/ckpt.pt \
        --out_dir ./out/stage2 \
        --epochs 10000 \
        --learning_rate 1e-4 \
        --batch_size 32 \
        --warmup_iters 500 \
        --eval_interval 500 \
        --device cuda \
        --clear_checkpoint
    '''
    run_command(cmd, "STAGE 2: Fine-tuning on Shakespeare")


def stage3_conversation_finetuning():
    """Stage 3: Fine-tune on Conversations (15-30 min)"""
    cmd = f'''"{PYTHON}" finetune_chatbot.py \
        --data ./data/Conversation.csv \
        --tokenizer {TOKENIZER} \
        --pretrained ./out/stage2/ckpt.pt \
        --out ./out/stage3 \
        --iters 5000 \
        --lr 5e-5 \
        --batch-size 64 \
        --eval-interval 250 \
        --device cuda
    '''
    run_command(cmd, "STAGE 3: Fine-tuning on Conversations")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Three-stage training pipeline for Scrappy LLM"
    )
    parser.add_argument(
        '--stage',
        choices=['1', '2', '3', 'all'],
        default='all',
        help='Which stage to run (default: all)'
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  THREE-STAGE TRAINING PIPELINE")
    print("="*70)
    print(f"  Tokenizer: {TOKENIZER}")
    print(f"  Stage selection: {args.stage}")
    print("="*70 + "\n")

    # Verify tokenizer exists
    if not os.path.exists(TOKENIZER):
        print(f"X ERROR: Tokenizer not found at {TOKENIZER}")
        print("Run: python build_standard_tokenizer.py first")
        sys.exit(1)

    if args.stage in ['1', 'all']:
        stage1_base_training()

    if args.stage in ['2', 'all']:
        # Verify stage 1 checkpoint exists
        if not os.path.exists('./out/stage1/ckpt.pt'):
            print("\nX ERROR: Stage 1 checkpoint not found")
            print("Run: python train_three_stage.py --stage 1 first")
            sys.exit(1)
        stage2_shakespeare_finetuning()

    if args.stage in ['3', 'all']:
        # Verify stage 2 checkpoint exists
        if not os.path.exists('./out/stage2/ckpt.pt'):
            print("\nX ERROR: Stage 2 checkpoint not found")
            print("Run: python train_three_stage.py --stage 2 first")
            sys.exit(1)
        stage3_conversation_finetuning()

    print("\n" + "="*70)
    print("  Checkmark PIPELINE COMPLETE")
    print("="*70)
    print("\nCheckpoints saved:")
    print("  - Stage 1 (Base): ./out/stage1/ckpt.pt")
    print("  - Stage 2 (Shakespeare): ./out/stage2/ckpt.pt")
    print("  - Stage 3 (Chatbot): ./out/stage3/chatbot_best.pt")
    print("\nTest the chatbot:")
    print(f'  "{PYTHON}" chatbot_inference.py --checkpoint ./out/stage3/chatbot_best.pt')


if __name__ == '__main__':
    main()
