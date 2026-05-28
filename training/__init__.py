"""
Training module for Scrappy LLM.
Contains model architecture, training utilities, and data handling.
"""

# Model architecture
from .model import (
    GPT,
    GPTBlock,
    RMSNorm,
    RoPEAttention,
    SwiGLU,
    ModelConfig,
    LlamaModel,
    LlamaBlock,
    precompute_rope_freqs,
    get_rotary_matrix
)

# Configuration
from .config import Config

# Tokenization
from .tokenizer import Tokenizer, CharTokenizer

# Data handling
from .data_loader import get_batch, load_char_data, load_or_create_data
from .data_prep import (
    train_char_vocab,
    create_bin_from_text,
    prepare_dataset,
    encode_to_uint16,
    load_tokenized_data
)

# Training
from .train import (
    train,
    build_optimizer,
    get_training_lr,
    estimate_loss,
    save_checkpoint,
    generate_text
)

# Fine-tuning
from .sft_trainer import (
    sft_train,
    load_pretrained_checkpoint,
    build_sft_dataset,
    get_sft_batch,
    save_sft_checkpoint
)

__all__ = [
    # Model
    'GPT',
    'GPTBlock',
    'RMSNorm',
    'RoPEAttention',
    'SwiGLU',
    'ModelConfig',
    'LlamaModel',
    'LlamaBlock',
    'precompute_rope_freqs',
    'get_rotary_matrix',

    # Config
    'Config',

    # Tokenization
    'Tokenizer',
    'CharTokenizer',

    # Data
    'get_batch',
    'load_char_data',
    'load_or_create_data',
    'train_char_vocab',
    'create_bin_from_text',
    'prepare_dataset',
    'encode_to_uint16',
    'load_tokenized_data',

    # Training
    'train',
    'build_optimizer',
    'get_training_lr',
    'estimate_loss',
    'save_checkpoint',
    'generate_text',

    # Fine-tuning
    'sft_train',
    'load_pretrained_checkpoint',
    'build_sft_dataset',
    'get_sft_batch',
    'save_sft_checkpoint',
]
