# Three-Stage Training Commands

## Prerequisites

Ensure the tokenizers library is installed:
```bash
"C:\Program Files\Python311\python.exe" -m pip install tokenizers
```

Verify your tokenizer exists:
```bash
ls out/standard_tokenizer.json
# Should show: 522KB file
```

## Option 1: Run All Stages (Recommended)

```bash
"C:\Program Files\Python311\python.exe" train_three_stage.py --stage all
```

**Total Time**: ~5-9 hours (mostly Stage 1)

## Option 2: Run Stages Individually

### Stage 1: Base Model Training (4-8 hours)

Train on TinyStories (2.1GB) to learn general language patterns:

```bash
"C:\Program Files\Python311\python.exe" train_three_stage.py --stage 1
```

Or manually:
```bash
"C:\Program Files\Python311\python.exe" -m training.train_scrappy ^
    --data ./data/TinyStoriesV2-GPT4-train.txt ^
    --tokenizer ./out/standard_tokenizer.json ^
    --out_dir ./out/stage1 ^
    --epochs 50000 ^
    --learning_rate 3e-4 ^
    --batch_size 32 ^
    --context_window 256 ^
    --d_model 256 ^
    --n_layer 6 ^
    --n_head 8 ^
    --warmup_iters 2000 ^
    --eval_interval 1000 ^
    --device cuda ^
    --clear_checkpoint
```

**Output**: `./out/stage1/ckpt.pt`

### Stage 2: Shakespeare Fine-tuning (30-60 minutes)

Fine-tune on Shakespeare (1.2MB) to learn Shakespearean style:

```bash
"C:\Program Files\Python311\python.exe" train_three_stage.py --stage 2
```

Or manually:
```bash
"C:\Program Files\Python311\python.exe" -m training.train_scrappy ^
    --data ./data/tinyshakespeare.txt ^
    --tokenizer ./out/standard_tokenizer.json ^
    --pretrained ./out/stage1/ckpt.pt ^
    --out_dir ./out/stage2 ^
    --epochs 10000 ^
    --learning_rate 1e-4 ^
    --batch_size 32 ^
    --warmup_iters 500 ^
    --eval_interval 500 ^
    --device cuda ^
    --clear_checkpoint
```

**Output**: `./out/stage2/ckpt.pt`

### Stage 3: Conversation Fine-tuning (15-30 minutes)

Fine-tune on conversations (259KB) to learn Q&A patterns:

```bash
"C:\Program Files\Python311\python.exe" train_three_stage.py --stage 3
```

Or manually:
```bash
"C:\Program Files\Python311\python.exe" finetune_chatbot.py ^
    --data ./data/Conversation.csv ^
    --tokenizer ./out/standard_tokenizer.json ^
    --pretrained ./out/stage2/ckpt.pt ^
    --out ./out/stage3 ^
    --iters 5000 ^
    --lr 5e-5 ^
    --batch-size 64 ^
    --eval-interval 250 ^
    --device cuda
```

**Output**: `./out/stage3/chatbot_best.pt`

## Testing After Each Stage

### Test Stage 1 (TinyStories)

```bash
"C:\Program Files\Python311\python.exe" -m training.train_scrappy ^
    --pretrained ./out/stage1/ckpt.pt ^
    --tokenizer ./out/standard_tokenizer.json ^
    --generate ^
    --prompt "Once upon a time"
```

**Expected**: Simple coherent story generation

### Test Stage 2 (Shakespeare)

```bash
"C:\Program Files\Python311\python.exe" -m training.train_scrappy ^
    --pretrained ./out/stage2/ckpt.pt ^
    --tokenizer ./out/standard_tokenizer.json ^
    --generate ^
    --prompt "To be or not to be"
```

**Expected**: Shakespearean language style

### Test Stage 3 (Chatbot)

```bash
"C:\Program Files\Python311\python.exe" chatbot_inference.py ^
    --checkpoint ./out/stage3/chatbot_best.pt ^
    --test
```

**Expected**: Conversational Q&A responses

## Hyperparameters Summary

| Stage | Data | Iterations | LR | Warmup | Batch | Time |
|-------|------|------------|-----|--------|-------|------|
| 1 | 2.1GB | 50,000 | 3e-4 | 2000 | 32 | 4-8h |
| 2 | 1.2MB | 10,000 | 1e-4 | 500 | 32 | 30-60m |
| 3 | 259KB | 5,000 | 5e-5 | 200 | 64 | 15-30m |

**Model**: 8192 vocab, 256 context, 256 embd, 6 layers, 8 heads (~6-8M params)

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
# For Stage 1
--batch_size 16

# For Stage 3
--batch-size 32
```

### Tokenizer Mismatch Error

Verify all stages use the same tokenizer:
```bash
ls out/standard_tokenizer.json
```

### Training Too Slow

Reduce iterations for testing:
```bash
# Stage 1: Use 20,000 instead of 50,000
--epochs 20000

# Stage 2: Use 5,000 instead of 10,000
--epochs 5000

# Stage 3: Use 2,000 instead of 5,000
--iters 2000
```

## Verification

Check checkpoints created:
```bash
ls -lh out/stage1/ckpt.pt
ls -lh out/stage2/ckpt.pt
ls -lh out/stage3/chatbot_best.pt
```

Verify vocab size in checkpoint:
```python
import torch
ckpt = torch.load('out/stage1/ckpt.pt', weights_only=False)
print(f"Vocab size: {ckpt['config']['vocab_size']}")
# Should print: 8192
```

## Quick Start (Recommended)

1. Install dependencies:
   ```bash
   "C:\Program Files\Python311\python.exe" -m pip install tokenizers
   ```

2. Start training:
   ```bash
   "C:\Program Files\Python311\python.exe" train_three_stage.py --stage 1
   ```

3. Let Stage 1 run overnight (4-8 hours)

4. Next day, run Stage 2:
   ```bash
   "C:\Program Files\Python311\python.exe" train_three_stage.py --stage 2
   ```

5. Finally, run Stage 3:
   ```bash
   "C:\Program Files\Python311\python.exe" train_three_stage.py --stage 3
   ```

6. Test your chatbot:
   ```bash
   "C:\Program Files\Python311\python.exe" chatbot_inference.py --checkpoint ./out/stage3/chatbot_best.pt
   ```
