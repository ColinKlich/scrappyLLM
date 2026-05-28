
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ----- Run identity ---
    run_name: str = "scrappy-v1"
    out_dir: str = "out"
    
    # ----- Data ---
    train_bin: str = "train.bin"
    val_bin: str = "val.bin"
    tokenizer_path: str = "scrappy_bpe.json"
    
    # ----- Model architecture ---
    vocab_size: int = 8192
    context_window: int = 64
    n_layer: int = 4
    n_head: int = 8
    d_model: int = 128
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # LR schedule
    warmup_iters: int = 200
    lr_decay_iters: int = 20000
    min_lr: float = 3e-5
    
    # Training loop
    batch_size: int = 32
    max_iters: int = 10000
    eval_interval: int = 100
    eval_iters: int = 10
    log_interval: int = 10
    
    # System
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 1337
    
    # SFT
    sft_dataset_path: Optional[str] = None
    sft_lr: float = 2e-5
    
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            run_name=d.get('run_name', cls.run_name),
            out_dir=d.get('out_dir', cls.out_dir),
            vocab_size=d.get('vocab_size', cls.vocab_size),
            context_window=d.get('context_window', cls.context_window),
            n_layer=d.get('n_layer', cls.n_layer),
            n_head=d.get('n_head', cls.n_head),
            d_model=d.get('d_model', cls.d_model),
            batch_size=d.get('batch_size', cls.batch_size),
            max_iters=d.get('epochs', d.get('max_iters', cls.max_iters)),
            learning_rate=d.get('learning_rate', cls.learning_rate),
            device=d.get('device', cls.device),
            dtype=d.get('dtype', cls.dtype),
            seed=d.get('seed', cls.seed),
        )
