"""
Query and inference utilities for Scrappy LLM.
Enhanced generation with top-k sampling, temperature control, and caching.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

# Try to import from training module, fallback to direct import for backward compatibility
try:
    from training import CharTokenizer, Tokenizer, GPT, ModelConfig
except ImportError:
    # For when query.py is used standalone
    pass


def generate_with_cache(
    model,
    ctx,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    config = None
) -> str:
    """Generate text with KV cache support for efficient sampling.
    
    Args:
        model: Trained GPT model
        ctx: Execution context (nullcontext or autocast)
        tokenizer: Tokenizer instance
        prompt: Input prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling (None = disabled)
        config: Configuration
        
    Returns:
        Generated text string
    """
    model.eval()

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_ids = list(prompt_ids)
    
    for _ in range(max_new_tokens):
        # Crop to context window
        block_size = model.config.block_size if hasattr(model, 'config') else 128
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Forward pass
        with ctx:
            logits, _ = model(idx_cond)
        
        # Get logits for next token
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        generated_ids.append(next_idx.item())
        idx = torch.cat((idx, next_idx), dim=1)
        
        # Stop at EOS if available
        if hasattr(tokenizer, 'eot_id') and next_idx.item() == tokenizer.eot_id:
            break
    
    return tokenizer.decode(generated_ids)


@torch.no_grad()
def generate_simple(
    model,
    tokenizer,
    prompt: str = "Hello, ",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 20,
    num_samples: int = 1
) -> list[str]:
    """Simple generation function for quick inference.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Starting prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        num_samples: Number of generations
        
    Returns:
        List of generated strings
    """
    model.eval()
    
    # Encode prompt
    if isinstance(prompt, str):
        prompt_ids = tokenizer.encode(prompt)
    else:
        prompt_ids = prompt
    
    results = []
    device = next(model.parameters()).device

    for sample_idx in range(num_samples):
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        block_size = model.config.block_size if hasattr(model, 'config') else 128
        
        for _ in range(max_new_tokens):
            # Crop context
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            
            # Forward
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        
        decoded = tokenizer.decode(idx[0].cpu().tolist())
        results.append(decoded)
    
    return results


def query_model(
    model,
    tokenizer,
    user_query: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 40,
    system_prompt: Optional[str] = None
) -> str:
    """Query the model with a user input (chat-like interface).
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        user_query: User's input query
        max_tokens: Maximum tokens to generate in response
        temperature: Sampling temperature
        top_k: Top-k sampling
        system_prompt: Optional system/instruction prefix
        
    Returns:
        Model's response (string)
    """
    prompt = user_query
    
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{user_query}\n"
    else:
        full_prompt = f"Query: {user_query}\nResponse: "
    
    results = generate_simple(
        model, tokenizer, full_prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        num_samples=1
    )
    
    if results:
        return results[0].replace(full_prompt, "").strip()
    return "Error generating response"


def load_and_query(
    model_path: str,
    tokenizer_path: str,
    query: str,
    device: str = 'auto'
) -> str:
    """Load model from checkpoint and generate response.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        tokenizer_path: Path to tokenizer JSON file
        query: User query
        device: Device to use
        
    Returns:
        Generated response
    """
    import torch
    
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config_data = checkpoint['config']
    
    # Create model
    from training import GPT, ModelConfig, Tokenizer

    # Build ModelConfig from checkpoint config
    if isinstance(config_data, dict):
        model_config = ModelConfig(
            vocab_size=config_data.get('vocab_size', 8192),
            block_size=config_data.get('context_window', config_data.get('block_size', 128)),
            n_layer=config_data.get('n_layer', 4),
            n_head=config_data.get('n_head', 8),
            n_embd=config_data.get('d_model', config_data.get('n_embd', 128))
        )
    else:
        model_config = ModelConfig(
            vocab_size=getattr(config_data, 'vocab_size', 8192),
            block_size=getattr(config_data, 'context_window', getattr(config_data, 'block_size', 128)),
            n_layer=getattr(config_data, 'n_layer', 4),
            n_head=getattr(config_data, 'n_head', 8),
            n_embd=getattr(config_data, 'd_model', getattr(config_data, 'n_embd', 128))
        )
    
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load tokenizer
    tokenizer = Tokenizer(tokenizer_path)
    
    # Generate
    result = query_model(model, tokenizer, query, max_tokens=100)
    
    return result


def generate_response(
    model,
    tokenizer,
    user_input: str,
    config,
    max_new_tokens: int = 30
) -> str:
    """Generate response to user input.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        user_input: User's text input
        config: Configuration (kept for backward compatibility)
        max_new_tokens: Max tokens to generate
        
    Returns:
        Generated response string
    """
    # Encode user input as initial sequence
    idx = tokenizer.encode(user_input)
    idx_tensor = torch.tensor(idx, dtype=torch.long, device=model.device).unsqueeze(0)
    
    generated = []
    
    for _ in range(max_new_tokens):
        # Get model output
        logits = model(idx_tensor)
        logits = logits[:, -1, :]
        
        # Sample
        probs = F.softmax(logits / 1.0, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        generated.append(next_token.item())
        idx_tensor = torch.cat([idx_tensor, next_token], dim=1)
    
    # Decode
    response = tokenizer.decode([idx_tensor[0, i].item() for i in range(idx_tensor.size(1))])
    return response
