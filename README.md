# Scrappy LLM

A modular PyTorch implementation of a GPT-style language model trained in four stages: base pretraining on TinyStories, literature fine-tuning, question-answering training, conversation fine-tuning, and optional Shakespeare style polish.

## Model Details

| Field | Value |
|-------|-------|
| **Architecture** | Decoder-only Transformer (GPT-style) |
| **Parameters** | ~50M (12 layers, 512 dims) |
| **Layers** | 12 (6x larger than original 6-layer model) |
| **Heads** | 8 |
| **Embedding dim** | 512 (original: 256) |
| **Context length** | 256 tokens |
| **Vocab size** | 8192 (custom BPE) |
| **Bias in Linear/LN** | False |
| **Tokenizer** | standard_tokenizer.json (HuggingFace BPE) |

## Training

**Four-stage training pipeline (improved for better QA and reduced overfitting):**

1. **Stage 1 - Base Pretraining**: TinyStoriesV2-GPT4 dataset (2.1GB)
   - Epochs: 20,000 (reduced from 50,000 to prevent overfitting)
   - Learning rate: 3e-4
   - Batch size: 32
   - Warmup: 2,000 iterations
   - Time: 6-10 hours on CUDA

2. **Stage 1b - Literature Continuation**: literature-condensed.txt (27MB)
   - Epochs: 5,000
   - Learning rate: 3e-4
   - Batch size: 32
   - Time: 1-2 hours on CUDA
   - Adds sophisticated vocabulary and literary knowledge

3. **Stage 2 - Question-Answering**: Natural Questions dataset (73MB)
   - Epochs: 8,000
   - Learning rate: 3e-5 (lower for fine-tuning)
   - Batch size: 32
   - Warmup: 500 iterations
   - Time: 1-2 hours on CUDA
   - Uses NQ-train_pairs.txt and NQ-open-train.txt

4. **Stage 3 - Conversation Fine-tuning**: Conversation.csv (259KB)
   - Epochs: 2,000 (reduced from 5,000)
   - Learning rate: 1e-5 (very low to preserve QA ability)
   - Batch size: 64
   - Time: 15-30 minutes on CUDA

5. **Stage 4 - Shakespeare Style (Optional)**: tinyshakespeare.txt (1.2MB)
   - Epochs: 1,000 (reduced from 10,000)
   - Learning rate: 3e-5 (very low - final polish)
   - Batch size: 32
   - Time: 10-20 minutes on CUDA
   - Can be skipped if style polish is not needed

**Key Improvements:**
- 6x larger model (50M vs 8.4M parameters)
- 283x more QA training data (73MB vs 259KB)
- Reduced overfitting with fewer epochs
- Shakespeare moved to end as optional style polish (won't degrade QA ability)

**Optimizer**: AdamW, weight_decay=0.1, betas=(0.9, 0.95), grad_clip=1.0

**LR schedule**: Linear warmup → cosine decay to min_lr

**Hardware/dtype**: Trained on CUDA (GPU), float32

## Quick Start

### Prerequisites

```bash
# Clone the repository
git clone <your-repo-url>
cd scrappyLLM

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Download and prepare the training data:

```bash
# Create data directory
mkdir -p data

# Download TinyStories dataset
# Place TinyStoriesV2-GPT4-train.txt in ./data/

# Download literature data
# Place literature-condensed.txt in ./data/

# Download Natural Questions dataset
# Run preprocessing:
python preprocess_nq_data.py
# This converts JSONL files to training format:
# - NQ-train_pairs.jsonl → NQ-train_pairs.txt
# - NQ-open.train.jsonl → NQ-open-train.txt
# - NQ-open.dev.jsonl → NQ-open-dev.txt

# Prepare conversation data
# Ensure Conversation.csv is in ./data/

# Optional: Shakespeare (for Stage 4)
# Ensure tinyshakespeare.txt is in ./data/
```

### Build the Tokenizer

**Important**: Build the tokenizer first before any training.

```bash
python build_standard_tokenizer.py \
  --vocab-size 8192 \
  --output ./out/standard_tokenizer.json \
  --data ./data/TinyStoriesV2-GPT4-train.txt ./data/tinyshakespeare.txt ./data/Conversation.csv
```

This creates a BPE tokenizer with 8192 tokens trained on all three datasets, ensuring consistent vocabulary across all training stages.

## Training

### Option 1: Full Four-Stage Pipeline (Recommended)

Run all stages sequentially:

```bash
# Run all stages INCLUDING Shakespeare (10-15 hours total)
python train_four_stage.py

# Or SKIP Shakespeare style polish (saves 10-20 min)
python train_four_stage.py --stage skip-shakespeare
```

Or run individual stages:

```bash
# Stage 1: Base training (6-10 hours)
python train_four_stage.py --stage 1

# Stage 1b: Literature (1-2 hours)
python train_four_stage.py --stage 1b

# Stage 2: Question-Answering (1-2 hours)
python train_four_stage.py --stage 2

# Stage 3: Conversation fine-tuning (15-30 min)
python train_four_stage.py --stage 3

# Stage 4: Shakespeare style - OPTIONAL (10-20 min)
python train_four_stage.py --stage 4
```

### Option 2: Manual Training (Advanced)

**Stage 1 - Base Pretraining:**

```bash
python -m training.train_scrappy \
  --data ./data/TinyStoriesV2-GPT4-train.txt \
  --tokenizer ./out/standard_tokenizer.json \
  --out_dir ./out/stage1 \
  --epochs 20000 \
  --learning_rate 3e-4 \
  --batch_size 32 \
  --context_window 256 \
  --d_model 512 \
  --n_layer 12 \
  --n_head 8 \
  --warmup_iters 2000 \
  --eval_interval 1000 \
  --device cuda
```

**Stage 1b - Literature Continuation:**

```bash
python -m training.train_scrappy \
  --data ./data/literature-condensed.txt \
  --tokenizer ./out/standard_tokenizer.json \
  --pretrained ./out/stage1/ckpt.pt \
  --out_dir ./out/stage1b_literature \
  --epochs 5000 \
  --learning_rate 3e-4 \
  --batch_size 32 \
  --warmup_iters 500 \
  --eval_interval 500 \
  --device cuda
```

**Stage 2 - Question-Answering:**

```bash
python -m training.train_scrappy \
  --data ./data/NQ-train_pairs.txt \
  --tokenizer ./out/standard_tokenizer.json \
  --pretrained ./out/stage1b_literature/ckpt.pt \
  --out_dir ./out/stage2_qa \
  --epochs 8000 \
  --learning_rate 3e-5 \
  --batch_size 32 \
  --warmup_iters 500 \
  --eval_interval 500 \
  --device cuda
```

**Stage 3 - Conversation Fine-tuning:**

```bash
python finetune_chatbot.py \
  --data ./data/Conversation.csv \
  --tokenizer ./out/standard_tokenizer.json \
  --pretrained ./out/stage2_qa/ckpt.pt \
  --out ./out/stage3_conversation \
  --iters 2000 \
  --lr 1e-5 \
  --batch-size 64 \
  --eval-interval 250 \
  --device cuda
```

**Stage 4 - Shakespeare Style (Optional):**

```bash
python -m training.train_scrappy \
  --data ./data/tinyshakespeare.txt \
  --tokenizer ./out/standard_tokenizer.json \
  --pretrained ./out/stage3_conversation/chatbot_best.pt \
  --out_dir ./out/stage4_shakespeare \
  --epochs 1000 \
  --learning_rate 3e-5 \
  --batch_size 32 \
  --warmup_iters 200 \
  --eval_interval 200 \
  --device cuda
```

## Inference

### Interactive Chatbot

Run the chatbot in interactive mode:

```bash
# If you ran all stages (with Shakespeare)
python chatbot_inference.py \
  --checkpoint ./out/stage4_shakespeare/ckpt.pt \
  --tokenizer ./out/standard_tokenizer.json

# If you skipped Shakespeare
python chatbot_inference.py \
  --checkpoint ./out/stage3_conversation/chatbot_best.pt \
  --tokenizer ./out/standard_tokenizer.json
```

Example conversation:

```
You: hello, how are you?
Bot: i'm fine. how about yourself?

You: what's your name?
Bot: my name is william.

You: quit
```

### Testing with Predefined Questions

```bash
python chatbot_inference.py \
  --checkpoint ./out/stage3_conversation/chatbot_best.pt \
  --tokenizer ./out/standard_tokenizer.json \
  --test
```

### Programmatic Generation

```python
from training import GPT, ModelConfig, Tokenizer
from query import generate_simple
import torch

# Load model and tokenizer
checkpoint = torch.load('./out/stage3_conversation/chatbot_best.pt')
config = checkpoint['config']

model_config = ModelConfig(
    vocab_size=config['vocab_size'],
    block_size=config['block_size'],
    n_layer=config['n_layer'],
    n_head=config['n_head'],
    n_embd=config['n_embd']
)

model = GPT(model_config)
model.load_state_dict(checkpoint['model'])
model.eval()

tokenizer = Tokenizer.auto_load('./out/standard_tokenizer.json')

# Generate text
prompt = "Q: hello, how are you?\nA:"
results = generate_simple(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_k=40
)

print(results[0])
```

## Project Structure

```
scrappyLLM/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── build_standard_tokenizer.py   # Build BPE tokenizer
├── train_four_stage.py           # Automated 4-stage pipeline (improved)
├── finetune_chatbot.py           # Stage 3 conversation fine-tuning
├── chatbot_inference.py          # Interactive chatbot
├── preprocess_nq_data.py         # Convert Natural Questions JSONL to text
├── query.py                      # Text generation utilities
│
├── data/                         # Training datasets
│   ├── TinyStoriesV2-GPT4-train.txt  (~2.1GB)
│   ├── literature-condensed.txt      (~27MB)
│   ├── NQ-train_pairs.txt           (~65MB preprocessed)
│   ├── NQ-open-train.txt            (~8.2MB preprocessed)
│   ├── NQ-open-dev.txt              (~383KB preprocessed)
│   ├── Conversation.csv              (~259KB)
│   └── tinyshakespeare.txt           (~1.2MB, optional)
│
├── out/                          # Output directory
│   ├── standard_tokenizer.json   # BPE tokenizer
│   ├── stage1/ckpt.pt           # Base model checkpoint
│   ├── stage1b_literature/ckpt.pt # Literature-tuned model
│   ├── stage2_qa/ckpt.pt        # QA-tuned model
│   ├── stage3_conversation/chatbot_best.pt # Conversation model
│   └── stage4_shakespeare/ckpt.pt # Optional Shakespeare-styled model
│
└── training/                     # Training module
    ├── __init__.py              # Module exports
    ├── config.py                # Configuration classes
    ├── model.py                 # GPT architecture
    ├── tokenizer.py             # Tokenizer classes
    ├── hf_tokenizer_wrapper.py # HuggingFace wrapper
    ├── train_scrappy.py         # Main training script
    ├── sft_trainer.py           # Supervised fine-tuning
    └── data_loader.py           # Data loading utilities
```

## Features

- **Multi-stage training**: Progressive 4-stage training from general language modeling to QA to conversation
- **Larger model**: 50M parameters (12 layers, 512 dims) for better language understanding
- **Better QA training**: 73MB of Natural Questions data (283x more than original conversation data)
- **Reduced overfitting**: Lower epoch counts to prevent catastrophic forgetting
- **Smart stage ordering**: QA before style training to prioritize reasoning over stylistic polish
- **Modern architecture**: RMSNorm, SwiGLU FFN, multi-head attention
- **Flexible tokenization**: BPE tokenizer with 8192 vocab size
- **Advanced training**: Warmup + cosine decay LR, gradient clipping, checkpointing
- **Easy inference**: Simple interactive chatbot interface
- **Modular design**: Clean separation between training, inference, and utilities

## Model Architecture Details

The model uses a decoder-only Transformer architecture inspired by GPT with some modern improvements:

- **Attention**: Multi-head self-attention (8 heads)
- **Normalization**: RMSNorm (more efficient than LayerNorm)
- **Activation**: SwiGLU in feed-forward layers (better than ReLU/GELU)
- **Positional encoding**: Learned positional embeddings
- **No bias**: All linear layers and normalization layers use `bias=False`

## Hyperparameter Guidelines

### Learning Rates
- **Base pretraining**: 3e-4 (standard GPT learning rate)
- **Domain fine-tuning**: 1e-4 (10x lower for stability)
- **Instruction fine-tuning**: 5e-5 (20x lower to preserve knowledge)

### Batch Sizes
- **Pretraining**: 32 (balance speed and memory)
- **Fine-tuning**: 32-64 (can increase for smaller datasets)

### Context Length
- **256 tokens**: Good balance between memory and coherence
- Can be adjusted in model config if needed

### Warmup
- **Base training**: 2000 steps (longer warmup for large dataset)
- **Fine-tuning**: 500 steps (shorter for small datasets)

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 16

# Reduce context window
--context_window 128

# Use gradient accumulation
--gradient_accumulation_steps 2
```

### Training is slow
- Ensure CUDA is available: `torch.cuda.is_available()`
- Use smaller dataset for testing
- Reduce evaluation frequency: `--eval-interval 2000`

### Poor generation quality
- Train longer (more epochs)
- Increase model size (more layers/dimensions)
- Improve data quality
- Adjust temperature (0.7-1.0 for diverse, 0.3-0.5 for focused)

### Tokenizer issues
- Rebuild tokenizer on all training data
- Ensure tokenizer is used consistently across all stages
- Check vocab size matches model config

## Performance Benchmarks

**Hardware**: NVIDIA GPU (CUDA-enabled)

| Stage | Dataset Size | Time | Notes |
|-------|-------------|------|-------|
| Stage 1 | 2.1GB | 6-10 hours | Base pretraining |
| Stage 1b | 27MB | 1-2 hours | Literature continuation |
| Stage 2 | 73MB | 1-2 hours | Question-Answering |
| Stage 3 | 259KB | 15-30 min | Conversation fine-tuning |
| Stage 4 | 1.2MB | 10-20 min | Shakespeare style (optional) |
| **Total** | ~2.2GB | **10-15 hours** | Full pipeline with Shakespeare |

**Note**: Actual times vary based on GPU model, batch size, and other settings.

## Comparison: Old vs New Pipeline

| Metric | Old (3-stage) | New (4-stage) | Improvement |
|--------|---------------|---------------|-------------|
| Parameters | 8.4M | 50M | 6x larger |
| Layers | 6 | 12 | 2x deeper |
| Embedding dims | 256 | 512 | 2x wider |
| QA Training Data | 259KB | 73MB | 283x more |
| Stage 1 Epochs | 50,000 | 20,000 | Less overfitting |
| Total Training Time | ~6 hours | ~12 hours | 2x longer |
| Shakespeare placement | Stage 2 | Stage 4 (optional) | Won't degrade QA |

## Citation

If you use this code, please cite:

```bibtex
@misc{scrappyllm2024,
  title={Scrappy LLM: A Modular GPT Implementation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/scrappyLLM}
}
```

## License

See LICENSE file for details.

## Acknowledgments

- **TinyStories**: Dataset from Microsoft Research
- **Shakespeare**: From Andrej Karpathy's char-rnn
- **Architecture**: Inspired by GPT, LLaMA, and nanoGPT

## Training Data

 - Training data pulled from [Hugging Face](https://huggingface.co/datasets/sentence-transformers/embedding-training-data)