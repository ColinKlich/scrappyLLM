# inference.py (updated for Shakespearean dialogues)
import torch
import json
import time
import argparse
import random
from llama import LlamaModel

# --- Device Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Configuration ---
MASTER_CONFIG = {
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 4,
    'context_window': 256,
    'vocab_size': None,
    'temperature': 0.8,
    'top_k': 50,
}

# Shakespearean characters for random selection
SHAKESPEAREAN_CHARACTERS = [
    "ROMEO", "JULIET", "HAMLET", "OPHELIA", "MACBETH", "LADY MACBETH",
    "LEAR", "IAGO", "OTHELLO", "DESDEMONA", "FALSTAFF", "MERCHANT",
    "MERCUTIO", "TYBALT", "BENVOLIO", "NURSE", "FRIAR LAURENCE",
    "PORTIA", "SHYLOCK", "PROSPERO", "ARIEL", "CALIBAN", "VIOLA",
    "ORSINO", "OLIVIA", "MALVOLIO", "BEATRICE", "BENEDICK",
    "PUCK", "OBERON", "TITANIA", "THESEUS", "HIPPOLYTA"
]

# Shakespearean greetings and prompts
SHAKESPEAREAN_GREETINGS = [
    "Hark! Who goes there?",
    "Good morrow, fair sir/madam!",
    "What news dost thou bring?",
    "Pray, speak thy mind.",
    "What wouldst thou have of me?",
    "By my troth, what brings thee hither?",
    "Well met! What tidings?",
    "I prithee, speak.",
]

# --- Load Vocabulary ---
def load_vocab(vocab_path='vocab.json'):
    """Load vocabulary from file."""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    stoi = vocab_data['stoi']
    itos = vocab_data['itos']
    MASTER_CONFIG['vocab_size'] = len(stoi)
    
    # Create encode/decode functions with loaded vocab
    def encode(s):
        return [stoi.get(c, 0) for c in s]
    
    def decode(l):
        return ''.join([itos.get(i, '') for i in l])
    
    return encode, decode, stoi, itos

# --- Generation Function ---
def generate_shakespearean_response(model, prompt, encode_func, decode_func, 
                                   max_new_tokens=200, temperature=None, top_k=None):
    """
    Generate a Shakespearean response to a user prompt.
    """
    model.eval()
    
    # Use config values if not specified
    if temperature is None:
        temperature = MASTER_CONFIG.get('temperature', 0.8)
    if top_k is None:
        top_k = MASTER_CONFIG.get('top_k', 50)
    
    # Format the prompt with special tokens if not already formatted
    if not prompt.startswith('<USER>:'):
        formatted_prompt = f"<USER>: {prompt} <BOT>: "
    else:
        formatted_prompt = prompt
    
    input_ids = torch.tensor(encode_func(formatted_prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # Use model's generate method
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    # Decode and extract only the response part
    full_text = decode_func(generated[0].tolist())
    
    # Extract response after <BOT>: 
    if '<BOT>:' in full_text:
        response = full_text.split('<BOT>: ')[-1]
    else:
        response = full_text
    
    # Remove <END> token and any trailing special tokens
    response = response.replace('<END>', '').strip()
    
    return response

# --- Interactive Chat Functions ---
def typing_effect(text, delay=0.03):
    """Print text with a typing effect."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def get_random_character():
    """Get a random Shakespearean character."""
    return random.choice(SHAKESPEAREAN_CHARACTERS)

def get_random_greeting():
    """Get a random Shakespearean greeting."""
    return random.choice(SHAKESPEAREAN_GREETINGS)

def chat_with_shakespeare_bot(model, encode_func, decode_func):
    """
    Interactive chat interface with the Shakespearean bot.
    """
    print("\n" + "="*60)
    print("SHAKESPEAREAN DIALOGUE BOT")
    print("="*60)
    print("Speak like a Shakespearean character!")
    print("Format your messages as: [CHARACTER] Your dialogue")
    print("Or just type your message for a random character.")
    print("\nCommands: 'quit' to exit, '!temp X', '!topk X', '!char' for new character")
    print("="*60 + "\n")
    
    # Current generation parameters
    current_temp = MASTER_CONFIG.get('temperature', 0.8)
    current_top_k = MASTER_CONFIG.get('top_k', 50)
    current_char = get_random_character()
    
    print(f"Current character: {current_char}")
    print(f"Settings: temperature={current_temp}, top_k={current_top_k}\n")
    
    # Start with a greeting
    greeting = get_random_greeting()
    print(f"{current_char}: {greeting}\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'farewell']:
                print(f"\n{current_char}: Fare thee well! Parting is such sweet sorrow...")
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.startswith('!temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        current_temp = new_temp
                        print(f"Temperature set to {current_temp}")
                    else:
                        print("Temperature must be between 0.1 and 2.0")
                except:
                    print("Invalid format. Use: !temp 0.8")
                continue
                
            elif user_input.startswith('!topk '):
                try:
                    new_top_k = int(user_input.split()[1])
                    if 0 <= new_top_k <= MASTER_CONFIG['vocab_size']:
                        current_top_k = new_top_k
                        print(f"Top-k set to {current_top_k}")
                    else:
                        print(f"Top-k must be between 0 and {MASTER_CONFIG['vocab_size']}")
                except:
                    print("Invalid format. Use: !topk 50")
                continue
                
            elif user_input.lower() in ['!char', '!character', '!new']:
                current_char = get_random_character()
                print(f"New character: {current_char}")
                continue
            
            # Format user input with character tag if not already present
            if not user_input.startswith('['):
                user_input = f"[{current_char}] {user_input}"
            
            # Generate response
            print(f"\n{current_char}: ", end='', flush=True)
            
            response = generate_shakespearean_response(
                model, 
                user_input,
                encode_func,
                decode_func,
                max_new_tokens=250,
                temperature=current_temp,
                top_k=current_top_k
            )
            
            # Print with typing effect
            typing_effect(response)
            
            # Update character if response starts with a different character
            if response.startswith('['):
                try:
                    new_char = response.split(']')[0][1:]
                    if new_char in SHAKESPEAREAN_CHARACTERS:
                        current_char = new_char
                        print(f"(Now speaking as {current_char})")
                except:
                    pass
            
        except KeyboardInterrupt:
            print(f"\n\n{current_char}: Parting is such sweet sorrow...")
            break
        except Exception as e:
            print(f"\nAlas, an error hath occurred: {e}")

# --- Load Model ---
def load_model(model_path="./models/llama_model.pth"):
    """Load the trained model."""
    print("Loading model...")
    
    # Create model with config
    model = LlamaModel(MASTER_CONFIG)
    
    # Load state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first or check the path.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model.to(device)
    model.eval()
    
    return model

# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shakespearean Dialogue Bot")
    parser.add_argument("--model", type=str, default="./models/llama_model.pth", 
                       help="Path to model weights")
    parser.add_argument("--vocab", type=str, default="vocab.json",
                       help="Path to vocabulary file")
    parser.add_argument("--test", action="store_true", 
                       help="Run test cases instead of interactive chat")
    parser.add_argument("--prompt", type=str, 
                       help="Single prompt to test (overrides --test)")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Top-k sampling value")
    
    args = parser.parse_args()
    
    # Load vocabulary
    try:
        encode_func, decode_func, stoi, itos = load_vocab(args.vocab)
        print(f"Vocabulary loaded with {len(stoi):,} tokens")
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {args.vocab}")
        print("Please train the model first to generate vocab.json")
        exit(1)
    
    # Load model
    model = load_model(args.model)
    if model is None:
        exit(1)
    
    # Handle different modes
    if args.prompt:
        # Single prompt mode
        response = generate_shakespearean_response(
            model,
            args.prompt,
            encode_func,
            decode_func,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"\nResponse: {response}")
    
    elif args.test:
        # Test mode with Shakespearean dialogues
        test_cases = [
            "[ROMEO] What light through yonder window breaks?",
            "[HAMLET] To be or not to be, that is the question.",
            "[JULIET] O Romeo, Romeo! Wherefore art thou Romeo?",
            "[MACBETH] Is this a dagger which I see before me?",
            "[LEAR] Blow, winds, and crack your cheeks!",
            "[IAGO] I am not what I am.",
            "[PORTIA] The quality of mercy is not strained.",
            "[FALSTAFF] The better part of valor is discretion.",
        ]
        
        print("\n" + "="*60)
        print("TESTING SHAKESPEAREAN DIALOGUES")
        print("="*60)
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\nTest {i}:")
            print(f"You: {prompt}")
            
            response = generate_shakespearean_response(
                model,
                prompt,
                encode_func,
                decode_func,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            
            print(f"Bot: {response}")
            
            if i < len(test_cases):
                print("-" * 40)
    
    else:
        # Interactive chat mode
        chat_with_shakespeare_bot(model, encode_func, decode_func)