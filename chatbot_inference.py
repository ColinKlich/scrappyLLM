#!/usr/bin/env python
"""
Chatbot inference script for interacting with fine-tuned model.
Interactive chat session with the trained chatbot.
"""

import torch
from training import GPT, ModelConfig, Tokenizer, CharTokenizer
from query import generate_simple
import os


def load_chatbot(checkpoint_path: str, tokenizer_path: str, device: str = "auto"):
    """Load trained chatbot model.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to use

    Returns:
        model, tokenizer
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading chatbot from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    config_data = checkpoint['config']

    # First, try to infer actual model dimensions from state dict
    model_state = checkpoint['model']
    actual_n_embd = None
    actual_n_layer = None
    actual_block_size = None

    if 'wte.weight' in model_state:
        # wte.weight shape is [vocab_size, n_embd]
        actual_n_embd = model_state['wte.weight'].shape[1]

    if 'wpe.weight' in model_state:
        # wpe.weight shape is [block_size, n_embd]
        actual_block_size = model_state['wpe.weight'].shape[0]

    # Count layers
    layer_indices = set()
    for key in model_state.keys():
        if key.startswith('layers.'):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    if layer_indices:
        actual_n_layer = max(layer_indices) + 1

    # Extract from config with fallbacks to inferred values
    if isinstance(config_data, dict):
        model_config = ModelConfig(
            vocab_size=config_data.get('vocab_size', 8192),
            block_size=actual_block_size or config_data.get('context_window', config_data.get('block_size', 128)),
            n_layer=actual_n_layer or config_data.get('n_layer', 4),
            n_head=config_data.get('n_head', 8),
            n_embd=actual_n_embd or config_data.get('d_model', config_data.get('n_embd', 128))
        )
    else:
        model_config = ModelConfig(
            vocab_size=getattr(config_data, 'vocab_size', 8192),
            block_size=actual_block_size or getattr(config_data, 'context_window', getattr(config_data, 'block_size', 128)),
            n_layer=actual_n_layer or getattr(config_data, 'n_layer', 4),
            n_head=getattr(config_data, 'n_head', 8),
            n_embd=actual_n_embd or getattr(config_data, 'd_model', getattr(config_data, 'n_embd', 128))
        )

    print(f"Model: {model_config.n_layer} layers, {model_config.n_embd} dims, {model_config.block_size} context")

    # Load model
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Loaded successfully: {model.num_parameters() / 1e6:.2f}M parameters")

    # Load tokenizer
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.auto_load(tokenizer_path)
    else:
        print("Warning: Tokenizer not found. Using character tokenizer.")
        # You'll need to have the original data to recreate it
        tokenizer = None

    return model, tokenizer


def chat_with_model(model, tokenizer, max_tokens: int = 80, temperature: float = 0.7, top_k: int = 40):
    """Interactive chat session.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
    """
    print("\n" + "="*60)
    print("Chatbot Ready! Type 'quit' or 'exit' to end the conversation.")
    print("="*60 + "\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Format prompt
        prompt = f"Q: {user_input}\nA:"

        # Generate response
        try:
            results = generate_simple(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                num_samples=1
            )

            if results:
                # Extract just the answer part
                response = results[0]

                # Try to extract answer after "A:"
                if "A:" in response:
                    response = response.split("A:", 1)[1].strip()
                    # Stop at next "Q:" if present
                    if "Q:" in response:
                        response = response.split("Q:")[0].strip()
                else:
                    response = response.replace(prompt, "").strip()

                # Clean up extra spaces between characters (BPE tokenizer artifact)
                # Replace multiple spaces with single space
                import re
                response = re.sub(r'\s+', ' ', response)

                print(f"Bot: {response}\n")
            else:
                print("Bot: [No response generated]\n")

        except Exception as e:
            print(f"Error: {e}\n")


def test_chatbot(model, tokenizer, test_questions: list = None):
    """Test chatbot with predefined questions.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_questions: List of questions to test
    """
    if test_questions is None:
        test_questions = [
            "hi, how are you doing?",
            "what school do you go to?",
            "how's it going?",
            "are you enjoying it there?",
            "so how have you been?",
        ]

    print("\n" + "="*60)
    print("Testing Chatbot")
    print("="*60 + "\n")

    for question in test_questions:
        prompt = f"Q: {question}\nA:"

        results = generate_simple(
            model,
            tokenizer,
            prompt,
            max_new_tokens=60,
            temperature=0.7,
            top_k=40,
            num_samples=1
        )

        if results:
            response = results[0].replace(prompt, "").strip()
            # Clean up the response
            if "\nQ:" in response:
                response = response.split("\nQ:")[0].strip()
        else:
            response = "[No response]"

        print(f"Q: {question}")
        print(f"A: {response}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive chatbot inference")
    parser.add_argument("--checkpoint", type=str, default="./out/chatbot_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="./out/standard_tokenizer.json",
                        help="Path to tokenizer")    
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/mps/auto)")
    parser.add_argument("--test", action="store_true",
                        help="Run test questions instead of interactive mode")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k sampling")
    parser.add_argument("--max-tokens", type=int, default=80,
                        help="Maximum tokens to generate")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_chatbot(args.checkpoint, args.tokenizer, args.device)

    if tokenizer is None:
        print("\nError: No tokenizer found. Cannot proceed.")
        print("Make sure tokenizer.json exists in the same directory as the checkpoint.")
        exit(1)

    # Run in test or interactive mode
    if args.test:
        test_chatbot(model, tokenizer)
    else:
        chat_with_model(model, tokenizer, args.max_tokens, args.temperature, args.top_k)
