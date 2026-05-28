# Scrappy LLM

A modular PyTorch implementation of a GPT-style language model trained in three stages: base pretraining on TinyStories, fine-tuning on Shakespeare, and instruction fine-tuning on conversational data.

## Model Details

| Field | Value |
|-------|-------|
| **Architecture** | Decoder-only Transformer (GPT-style) |
| **Parameters** | ~8.39M |
| **Layers** | 6 |
| **Heads** | 8 |
| **Embedding dim** | 256 |
| **Context length** | 256 tokens |
| **Vocab size** | 8192 (custom BPE) |
| **Bias in Linear/LN** | False |
| **Tokenizer** | standard_tokenizer.json (HuggingFace BPE) |

## Training

**Three-stage training pipeline:**

1. **Stage 1 - Base Pretraining**: TinyStoriesV2-GPT4 dataset (2.1GB)
   - Epochs: 50,000
   - Learning rate: 3e-4
   - Batch size: 32
   - Warmup: 2,000 iterations
   - Time: 4-8 hours on CUDA

2. **Stage 2 - Domain Fine-tuning**: tinyshakespeare.txt (1.2MB)
   - Epochs: 10,000
   - Learning rate: 1e-4 (lower for fine-tuning)
   - Batch size: 32
   - Warmup: 500 iterations
   - Time: 30-60 minutes on CUDA

3. **Stage 3 - Instruction Fine-tuning**: Conversation.csv (259KB)
   - Iterations: 5,000
   - Learning rate: 5e-5 (lowest for SFT)
   - Batch size: 64
   - Evaluation interval: 250 steps
   - Time: 15-30 minutes on CUDA

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

# Download Shakespeare (included in repo)
# Ensure tinyshakespeare.txt is in ./data/

# Prepare conversation data
# Ensure Conversation.csv is in ./data/
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

### Option 1: Full Three-Stage Pipeline (Recommended)

Run all three stages sequentially:

```bash
python train_three_stage.py --stage all
```

Or run individual stages:

```bash
# Stage 1: Base training (4-8 hours)
python train_three_stage.py --stage 1

# Stage 2: Shakespeare fine-tuning (30-60 min)
python train_three_stage.py --stage 2

# Stage 3: Conversation fine-tuning (15-30 min)
python train_three_stage.py --stage 3
```

### Option 2: Manual Training

**Stage 1 - Base Pretraining:**

```bash
python -m training.train_scrappy \
  --data ./data/TinyStoriesV2-GPT4-train.txt \
  --tokenizer ./out/standard_tokenizer.json \
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
  --device cuda
```

**Stage 2 - Shakespeare Fine-tuning:**

```bash
python -m training.train_scrappy \
  --data ./data/tinyshakespeare.txt \
  --tokenizer ./out/standard_tokenizer.json \
  --pretrained ./out/stage1/ckpt.pt \
  --out_dir ./out/stage2 \
  --epochs 10000 \
  --learning_rate 1e-4 \
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
  --pretrained ./out/stage2/ckpt.pt \
  --out ./out/stage3 \
  --iters 5000 \
  --lr 5e-5 \
  --batch-size 64 \
  --eval-interval 250 \
  --device cuda
```

## Inference

### Interactive Chatbot

Run the chatbot in interactive mode:

```bash
python chatbot_inference.py \
  --checkpoint ./out/stage3/chatbot_best.pt \
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
  --checkpoint ./out/stage3/chatbot_best.pt \
  --tokenizer ./out/standard_tokenizer.json \
  --test
```

### Programmatic Generation

```python
from training import GPT, ModelConfig, Tokenizer
from query import generate_simple
import torch

# Load model and tokenizer
checkpoint = torch.load('./out/stage3/chatbot_best.pt')
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
├── TRAINING_COMMANDS.md          # Additional training notes
│
├── build_standard_tokenizer.py   # Build BPE tokenizer
├── train_three_stage.py          # Automated 3-stage pipeline
├── finetune_chatbot.py           # Stage 3 conversation fine-tuning
├── chatbot_inference.py          # Interactive chatbot
├── query.py                      # Text generation utilities
│
├── data/                         # Training datasets
│   ├── TinyStoriesV2-GPT4-train.txt  (~2.1GB)
│   ├── tinyshakespeare.txt           (~1.2MB)
│   └── Conversation.csv              (~259KB)
│
├── out/                          # Output directory
│   ├── standard_tokenizer.json   # BPE tokenizer
│   ├── stage1/ckpt.pt           # Base model checkpoint
│   ├── stage2/ckpt.pt           # Shakespeare-tuned model
│   └── stage3/chatbot_best.pt   # Final chatbot model
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

- **Multi-stage training**: Progressive training from general language modeling to task-specific fine-tuning
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

| Stage | Dataset Size | Time | Final Loss |
|-------|-------------|------|------------|
| Stage 1 | 2.1GB | 4-8 hours | ~3.5-4.0 |
| Stage 2 | 1.2MB | 30-60 min | ~2.8-3.2 |
| Stage 3 | 259KB | 15-30 min | ~3.0-3.5 |

**Note**: Actual times vary based on GPU model, batch size, and other settings.

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
