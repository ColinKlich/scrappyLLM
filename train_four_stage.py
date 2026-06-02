#!/usr/bin/env python
"""
Five-stage training pipeline for Scrappy LLM (IMPROVED):
1. Base model on TinyStories (4-8 hours) - 12 layers, 512 dims (~50M params)
2. Continue on Literature (1-2 hours)
3. Fine-tune on NQ Question-Answering (1-2 hours)
4. Fine-tune on Conversations (15-30 min)
5. Fine-tune on Shakespeare (10-20 min) - STYLE POLISH ONLY

All stages use the same standard BPE tokenizer (8192 vocab).

IMPROVEMENTS:
- 6x larger model (50M vs 8.4M params)
- Added literature training (27MB classic books)
- Added QA training (73MB Natural Questions dataset)
- Reduced epochs to prevent overfitting
- Lower learning rates for fine-tuning stages
- Shakespeare moved to END (style layer, won't degrade QA ability)
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
        print(f"\n✗ {description} FAILED")
        sys.exit(1)
    print(f"\n✓ {description} COMPLETED")


def preprocess_nq_data():
    """Preprocess NQ datasets before training"""
    print("\n" + "="*70)
    print("  PREPROCESSING: Natural Questions Datasets")
    print("="*70)

    # Check if already preprocessed
    if (os.path.exists('./data/NQ-train_pairs.txt') and
        os.path.exists('./data/NQ-open-train.txt') and
        os.path.exists('./data/NQ-open-dev.txt')):
        print("✓ NQ datasets already preprocessed")
        return

    cmd = f'"{PYTHON}" preprocess_nq_data.py'
    run_command(cmd, "Preprocessing NQ Datasets")


def stage1_base_training():
    """Stage 1: Base model on TinyStories + Literature (6-10 hours)

    Model: 12 layers, 512 dims, 8 heads = ~50M parameters
    Context: 256 tokens
    Data: TinyStories (2.1GB) + Literature (27MB) = 2.13GB
    """
    cmd = f'''"{PYTHON}" -m training.train_scrappy \
        --data ./data/TinyStoriesV2-GPT4-train.txt \
        --val_data ./data/TinyStoriesV2-GPT4-valid.txt \
        --tokenizer {TOKENIZER} \
        --out_dir ./out/stage1_large \
        --epochs 20000 \
        --learning_rate 3e-4 \
        --batch_size 32 \
        --context_window 256 \
        --d_model 512 \
        --n_layer 12 \
        --n_head 8 \
        --warmup_iters 2000 \
        --eval_interval 1000 \
        --device cuda \
        --clear_checkpoint
    '''
    run_command(cmd, "STAGE 1: Base Training on TinyStories + Literature")


def stage1b_literature_continuation():
    """Stage 1b: Continue training on literature corpus

    This adds literary knowledge without forgetting TinyStories base.
    """
    cmd = f'''"{PYTHON}" -m training.train_scrappy \
        --data ./data/literature-condensed.txt \
        --tokenizer {TOKENIZER} \
        --pretrained ./out/stage1_large/ckpt.pt \
        --out_dir ./out/stage1b_literature \
        --epochs 5000 \
        --learning_rate 1e-4 \
        --batch_size 32 \
        --context_window 256 \
        --warmup_iters 500 \
        --eval_interval 500 \
        --device cuda \
        --clear_checkpoint
    '''
    run_command(cmd, "STAGE 1b: Literature Continuation Training")


def stage2_qa_finetuning():
    """Stage 2: Fine-tune on Natural Questions (1-2 hours)

    This stage teaches the model to answer factual questions.
    Uses combined NQ datasets (73MB total).
    Moved BEFORE Shakespeare to prioritize core reasoning over style.
    """
    # Create combined QA dataset
    print("\nCombining NQ datasets for training...")
    combined_path = './data/NQ-combined-train.txt'

    if not os.path.exists(combined_path):
        with open(combined_path, 'w', encoding='utf-8') as fout:
            # Add NQ-train_pairs
            if os.path.exists('./data/NQ-train_pairs.txt'):
                with open('./data/NQ-train_pairs.txt', 'r', encoding='utf-8') as fin:
                    fout.write(fin.read())
                print("  ✓ Added NQ-train_pairs.txt")

            # Add NQ-open-train
            if os.path.exists('./data/NQ-open-train.txt'):
                with open('./data/NQ-open-train.txt', 'r', encoding='utf-8') as fin:
                    fout.write(fin.read())
                print("  ✓ Added NQ-open-train.txt")

        print(f"✓ Combined dataset created: {combined_path}")

    # Validation data
    val_data_arg = '--val_data ./data/NQ-open-dev.txt' if os.path.exists('./data/NQ-open-dev.txt') else ''

    cmd = f'''"{PYTHON}" -m training.train_scrappy \
        --data {combined_path} \
        {val_data_arg} \
        --tokenizer {TOKENIZER} \
        --pretrained ./out/stage1b_literature/ckpt.pt \
        --out_dir ./out/stage2_qa \
        --epochs 8000 \
        --learning_rate 3e-5 \
        --batch_size 64 \
        --context_window 256 \
        --warmup_iters 500 \
        --eval_interval 500 \
        --device cuda \
        --clear_checkpoint
    '''
    run_command(cmd, "STAGE 2: Fine-tuning on Natural Questions")


def stage3_conversation_finetuning():
    """Stage 3: Fine-tune on Conversations (15-30 min)

    Teaches conversational turn-taking format.
    Very low learning rate (1e-5) to preserve QA knowledge.
    """
    cmd = f'''"{PYTHON}" finetune_chatbot.py \
        --data ./data/Conversation.csv \
        --tokenizer {TOKENIZER} \
        --pretrained ./out/stage2_qa/ckpt.pt \
        --out ./out/stage3_conversation \
        --iters 2000 \
        --lr 1e-5 \
        --batch-size 64 \
        --eval-interval 200 \
        --device cuda
    '''
    run_command(cmd, "STAGE 3: Fine-tuning on Conversations")


def stage4_shakespeare_finetuning():
    """Stage 4: Fine-tune on Shakespeare (10-20 min) - FINAL POLISH

    Moved to the END as a style layer that won't degrade QA/conversation abilities.
    Very low learning rate (3e-5) and reduced epochs (1000) since it's just style polish.
    This stage is OPTIONAL - skip if you don't need Shakespearean style.
    """
    cmd = f'''"{PYTHON}" -m training.train_scrappy \
        --data ./data/tinyshakespeare.txt \
        --tokenizer {TOKENIZER} \
        --pretrained ./out/stage3_conversation/chatbot_best.pt \
        --out_dir ./out/stage4_shakespeare \
        --epochs 1000 \
        --learning_rate 3e-5 \
        --batch_size 32 \
        --context_window 256 \
        --warmup_iters 100 \
        --eval_interval 200 \
        --device cuda \
        --clear_checkpoint
    '''
    run_command(cmd, "STAGE 4: Fine-tuning on Shakespeare (Style Polish)")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Four-stage training pipeline for Scrappy LLM (IMPROVED)"
    )
    parser.add_argument(
        '--stage',
        choices=['preprocess', '1', '1b', '2', '3', '4', 'all', 'skip-shakespeare'],
        default='all',
        help='Which stage to run (default: all, skip-shakespeare skips stage 4)'
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  FIVE-STAGE TRAINING PIPELINE (IMPROVED)")
    print("="*70)
    print(f"  Model: 12 layers, 512 dims (~50M parameters)")
    print(f"  Context: 256 tokens")
    print(f"  Tokenizer: {TOKENIZER}")
    print(f"  Stage selection: {args.stage}")
    print("="*70)
    print("\n  STAGE ORDER (optimized):")
    print("    1. TinyStories (base language)")
    print("    1b. Literature (vocabulary + knowledge)")
    print("    2. Natural Questions (QA reasoning)")
    print("    3. Conversations (conversational format)")
    print("    4. Shakespeare (style polish - OPTIONAL)")
    print("="*70 + "\n")

    # Verify tokenizer exists
    if not os.path.exists(TOKENIZER):
        print(f"✗ ERROR: Tokenizer not found at {TOKENIZER}")
        print("Run: python build_standard_tokenizer.py first")
        sys.exit(1)

    # Preprocess NQ data
    if args.stage in ['preprocess', 'all', 'skip-shakespeare']:
        preprocess_nq_data()

    if args.stage in ['1', 'all', 'skip-shakespeare']:
        stage1_base_training()

    if args.stage in ['1b', 'all', 'skip-shakespeare']:
        # Verify stage 1 checkpoint exists
        if not os.path.exists('./out/stage1_large/ckpt.pt'):
            print("\n✗ ERROR: Stage 1 checkpoint not found")
            print("Run: python train_four_stage.py --stage 1 first")
            sys.exit(1)
        stage1b_literature_continuation()

    if args.stage in ['2', 'all', 'skip-shakespeare']:
        # Verify stage 1b checkpoint exists
        if not os.path.exists('./out/stage1b_literature/ckpt.pt'):
            print("\n✗ ERROR: Stage 1b checkpoint not found")
            print("Run: python train_four_stage.py --stage 1b first")
            sys.exit(1)
        stage2_qa_finetuning()

    if args.stage in ['3', 'all', 'skip-shakespeare']:
        # Verify stage 2 checkpoint exists
        if not os.path.exists('./out/stage2_qa/ckpt.pt'):
            print("\n✗ ERROR: Stage 2 checkpoint not found")
            print("Run: python train_four_stage.py --stage 2 first")
            sys.exit(1)
        stage3_conversation_finetuning()

    if args.stage in ['4', 'all']:
        # Verify stage 3 checkpoint exists
        if not os.path.exists('./out/stage3_conversation/chatbot_best.pt'):
            print("\n✗ ERROR: Stage 3 checkpoint not found")
            print("Run: python train_four_stage.py --stage 3 first")
            sys.exit(1)
        stage4_shakespeare_finetuning()

    print("\n" + "="*70)
    print("  ✓ PIPELINE COMPLETE")
    print("="*70)
    print("\nCheckpoints saved:")
    print("  - Stage 1 (Base): ./out/stage1_large/ckpt.pt")
    print("  - Stage 1b (Literature): ./out/stage1b_literature/ckpt.pt")
    print("  - Stage 2 (QA): ./out/stage2_qa/ckpt.pt")
    print("  - Stage 3 (Conversation): ./out/stage3_conversation/chatbot_best.pt")
    if args.stage in ['4', 'all']:
        print("  - Stage 4 (Shakespeare): ./out/stage4_shakespeare/ckpt.pt")
    print("\nModel Stats:")
    print("  - Parameters: ~50M")
    print("  - Layers: 12")
    print("  - Hidden dim: 512")
    print("  - Context: 256 tokens")

    # Determine best checkpoint based on what was run
    if args.stage in ['4', 'all']:
        best_checkpoint = './out/stage4_shakespeare/ckpt.pt'
        stage_name = "Stage 4 (with Shakespeare style)"
    else:
        best_checkpoint = './out/stage3_conversation/chatbot_best.pt'
        stage_name = "Stage 3 (without Shakespeare)"

    print(f"\nBest checkpoint for inference: {stage_name}")
    print(f"  {best_checkpoint}")
    print("\nTest the chatbot:")
    print(f'  "{PYTHON}" chatbot_inference.py --checkpoint {best_checkpoint}')
    print("\nExport to safetensors:")
    print(f'  "{PYTHON}" export_safetensors.py --checkpoint {best_checkpoint} --output ./out/chatbot_best_large.safetensors')


if __name__ == '__main__':
    main()
