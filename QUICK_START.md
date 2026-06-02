# Quick Start: Improved Training Pipeline

## Prerequisites
```bash
# Verify you have these files
data/TinyStoriesV2-GPT4-train.txt     (2.1 GB)
data/TinyStoriesV2-GPT4-valid.txt     
data/literature-condensed.txt         (27 MB)
data/NQ-train_pairs.jsonl             (65 MB)
data/NQ-open.train.jsonl              (8.2 MB)
data/NQ-open.dev.jsonl                (383 KB)
data/tinyshakespeare.txt              (1.2 MB)
data/Conversation.csv                 (259 KB)
out/standard_tokenizer.json           (must exist)
```

## Step 1: Preprocess NQ Data
```bash
python preprocess_nq_data.py
```

This creates:
- `data/NQ-train_pairs.txt`
- `data/NQ-open-train.txt`
- `data/NQ-open-dev.txt`

## Step 2: Train the Model

### Option A: Full Training (WITH Shakespeare style)
```bash
python train_four_stage.py
```
**Time:** ~12-15 hours  
**Result:** Model that can do QA, conversations, AND Shakespeare style

### Option B: Skip Shakespeare (QA + Conversations only)
```bash
python train_four_stage.py --stage skip-shakespeare
```
**Time:** ~10-12 hours  
**Result:** Model that can do QA and conversations (no archaic style)

### Option C: Individual Stages
```bash
python train_four_stage.py --stage 1    # 6-10 hours
python train_four_stage.py --stage 1b   # 1-2 hours
python train_four_stage.py --stage 2    # 1-2 hours (QA)
python train_four_stage.py --stage 3    # 15-30 min (Conversations)
python train_four_stage.py --stage 4    # 10-20 min (Shakespeare - optional)
```

## Step 3: Test Your Model

### With Shakespeare:
```bash
python chatbot_inference.py --checkpoint ./out/stage4_shakespeare/ckpt.pt
```

### Without Shakespeare:
```bash
python chatbot_inference.py --checkpoint ./out/stage3_conversation/chatbot_best.pt
```

## Expected Results

### Before (Old Model):
```
You: tell me a joke
Bot: whataretheygoingtodo?

You: why is the sky blue?
Bot: i'mlookinglookingintheglass.
```

### After (New Model):
```
You: tell me a joke
Bot: Why did the chicken cross the road? To get to the other side!

You: why is the sky blue?
Bot: The sky appears blue because of Rayleigh scattering. Sunlight 
     enters Earth's atmosphere and shorter blue wavelengths scatter 
     more than longer wavelengths like red.
```

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: TinyStories (2.1GB)                            │
│ ├─ 12 layers, 512 dims, 50M params                      │
│ ├─ 20,000 epochs                                        │
│ └─ Time: 6-10 hours                                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 1b: Literature (27MB)                             │
│ ├─ Adds vocabulary + world knowledge                    │
│ ├─ 5,000 epochs, LR: 1e-4                               │
│ └─ Time: 1-2 hours                                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Natural Questions QA (73MB) ← MOVED UP!        │
│ ├─ Teaches factual reasoning                            │
│ ├─ 8,000 epochs, LR: 3e-5                               │
│ └─ Time: 1-2 hours                                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Conversations (259KB)                          │
│ ├─ Adds Q: A: conversational format                     │
│ ├─ 2,000 iters, LR: 1e-5                                │
│ └─ Time: 15-30 min                                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Shakespeare (1.2MB) ← MOVED TO END! (optional) │
│ ├─ Style polish only                                    │
│ ├─ 1,000 epochs, LR: 3e-5                               │
│ └─ Time: 10-20 min                                      │
└─────────────────────────────────────────────────────────┘
```

## Key Improvements

| Metric | Old | New | Change |
|--------|-----|-----|--------|
| Model size | 8.4M | 50M | 6x larger |
| QA training data | 259 KB | 73 MB | 283x more |
| Stage 1 epochs | 50,000 | 20,000 | Less overfitting |
| Shakespeare placement | Stage 2 | Stage 4 | Won't degrade QA |
| Shakespeare epochs | 10,000 | 1,000 | Less forgetting |
| Shakespeare LR | 1e-4 | 3e-5 | Gentler fine-tuning |

## Troubleshooting

### "Out of memory"
Reduce batch sizes in `train_four_stage.py`:
- Stage 1/1b/2: `--batch_size 16` (from 32)
- Stage 3: `--batch_size 32` (from 64)

### "Model still outputs gibberish"
1. Check tokenizer spacing bug in `chatbot_inference.py`
2. Verify validation loss is decreasing during training
3. Consider training longer (increase epochs)

### "I don't care about Shakespeare"
Run with `--stage skip-shakespeare` to save time!

## What's Next?

1. ✅ Run preprocessing: `python preprocess_nq_data.py`
2. ✅ Start training: `python train_four_stage.py`
3. ✅ Test model: `python chatbot_inference.py --checkpoint <path>`
4. ✅ Export to safetensors for deployment

Good luck! Your model should be much smarter with the larger architecture and better training data.
