# train.py
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import os
import json
import re
from llama import LlamaModel

# --- Device Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Configuration ---
MASTER_CONFIG = {
    'batch_size': 32,
    'context_window': 256,  # Increased for longer conversations
    'd_model': 128,
    'epochs': 10,
    'log_interval': 10,
    'n_heads': 8,
    'n_layers': 4,
    'temperature': 0.8,
    'top_k': 50,
}

# --- Data Loading & Preprocessing ---
print("Loading Shakespeare dataset...")
with open('./data/tinyshakespeare.txt', 'r') as f:
    shakespeare_text = f.read()

print("Parsing Shakespearean dialogues...")

def parse_shakespeare_dialogues(text):
    """
    Parse Shakespeare text into speaker-turn pairs.
    Format: CHARACTER: Dialogue
    """
    lines = text.strip().split('\n')
    dialogues = []
    current_speaker = None
    current_dialogue = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a character name (uppercase followed by colon)
        if re.match(r'^[A-Z][A-Z\s]+:', line):
            # If we have accumulated dialogue from previous speaker, save it
            if current_speaker and current_dialogue:
                dialogues.append((current_speaker, ' '.join(current_dialogue)))
            
            # Parse new speaker
            parts = line.split(':', 1)
            current_speaker = parts[0].strip()
            current_dialogue = [parts[1].strip()] if len(parts) > 1 else []
        else:
            # Continuation of current speaker's dialogue
            if current_dialogue is not None:
                current_dialogue.append(line)
    
    # Add the last speaker's dialogue
    if current_speaker and current_dialogue:
        dialogues.append((current_speaker, ' '.join(current_dialogue)))
    
    return dialogues

def create_conversation_pairs(dialogues):
    """
    Create conversation pairs from parsed dialogues.
    Each pair is (speaker1 dialogue, speaker2 response)
    """
    pairs = []
    
    for i in range(len(dialogues) - 1):
        speaker1, dialogue1 = dialogues[i]
        speaker2, dialogue2 = dialogues[i + 1]
        
        # Only create pairs if both dialogues have reasonable length
        if len(dialogue1) > 10 and len(dialogue2) > 10:
            # Format: <USER>: [speaker1] dialogue1 <BOT>: [speaker2] dialogue2
            user_input = f"[{speaker1}] {dialogue1}"
            bot_response = f"[{speaker2}] {dialogue2}"
            pairs.append((user_input, bot_response))
    
    return pairs

# Parse the dialogues
dialogues = parse_shakespeare_dialogues(shakespeare_text)
print(f"Parsed {len(dialogues)} dialogue turns")

# Create conversation pairs
dialogue_pairs = create_conversation_pairs(dialogues)
print(f"Created {len(dialogue_pairs)} conversation pairs")

# Display some examples
print("\nSample conversation pairs:")
for i, (user, bot) in enumerate(dialogue_pairs[:3]):
    print(f"\nPair {i + 1}:")
    print(f"  User: {user[:80]}...")
    print(f"  Bot:  {bot[:80]}...")

# Create vocabulary from all dialogues
all_text = []
for user_input, bot_response in dialogue_pairs:
    all_text.append(user_input)
    all_text.append(bot_response)

vocab = sorted(list(set(''.join(all_text))))
MASTER_CONFIG['vocab_size'] = len(vocab)

# Create mappings
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode(s):
    """Convert string to list of integers."""
    return [stoi.get(ch, 0) for ch in s]

def decode(l):
    """Convert list of integers to string."""
    return ''.join([itos.get(i, '') for i in l])

# Prepare dataset with proper formatting
dataset = []
for user_input, bot_response in dialogue_pairs:
    formatted_input = f"<USER>: {user_input} <BOT>: "
    formatted_output = bot_response + " <END>"
    
    dataset.append((
        torch.tensor(encode(formatted_input), dtype=torch.long),
        torch.tensor(encode(formatted_output), dtype=torch.long)
    ))

print(f"\nDataset size: {len(dataset)} pairs")
print(f"Vocabulary size: {MASTER_CONFIG['vocab_size']}")

# Analyze dataset lengths
input_lengths = [len(pair[0]) for pair in dataset]
output_lengths = [len(pair[1]) for pair in dataset]

print(f"Average input length: {np.mean(input_lengths):.1f}")
print(f"Average output length: {np.mean(output_lengths):.1f}")
print(f"Max input length: {max(input_lengths)}")
print(f"Max output length: {max(output_lengths)}")

# --- Data Batching ---
def get_batches(data, split='train', batch_size=None, context_window=None, config=MASTER_CONFIG):
    """Generate batches of data for training or validation."""
    batch_size = batch_size or config['batch_size']
    context_window = context_window or config['context_window']
    
    # Split data
    train_split = int(0.9 * len(data))
    train_data = data[:train_split]
    val_data = data[train_split:]
    
    batch_data = train_data if split == 'train' else val_data
    
    # Randomly select batch indices
    ix = torch.randint(0, len(batch_data), (batch_size,))
    
    # Pad sequences
    x = torch.nn.utils.rnn.pad_sequence(
        [batch_data[i][0] for i in ix], 
        batch_first=True, 
        padding_value=0
    )
    y = torch.nn.utils.rnn.pad_sequence(
        [batch_data[i][1] for i in ix], 
        batch_first=True, 
        padding_value=0
    )
    
    # Truncate or pad to context_window
    if x.shape[1] < context_window:
        x = F.pad(x, (0, context_window - x.shape[1]), 'constant', 0)
    else:
        x = x[:, :context_window]
        
    if y.shape[1] < context_window:
        y = F.pad(y, (0, context_window - y.shape[1]), 'constant', 0)
    else:
        y = y[:, :context_window]
    
    return x.to(device), y.to(device)

# --- Training & Evaluation ---
@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
    """Evaluate the model's loss on training and validation sets."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            _, loss = model(xb, targets=yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out

def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    """Train the language model."""
    losses = []
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])
        logits, loss = model(xs, targets=ys)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            eval_losses = evaluate_loss(model)
            losses.append(eval_losses)
            
            if print_logs:
                print(f"Epoch {epoch} | train loss {eval_losses['train']:.3f} | "
                      f"val loss {eval_losses['val']:.3f} | Time {batch_time:.3f}")
            
            start_time = time.time()
    
    print(f"Final validation loss: {losses[-1]['val']:.4f}")
    return pd.DataFrame(losses).plot()

# --- Text Generation Helper ---
def generate_shakespearean_response(model, prompt, max_new_tokens=200, config=MASTER_CONFIG):
    """Generate a Shakespearean response to a user prompt."""
    model.eval()
    
    # Format prompt with user tag
    if not prompt.startswith('<USER>:'):
        formatted_prompt = f"<USER>: {prompt} <BOT>: "
    else:
        formatted_prompt = prompt
    
    input_ids = torch.tensor(encode(formatted_prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=config.get('temperature', 0.8),
        top_k=config.get('top_k', 50)
    )
    
    model.train()
    
    # Decode and extract only the response part
    full_text = decode(generated[0].tolist())
    
    # Extract response after <BOT>: 
    if '<BOT>:' in full_text:
        response = full_text.split('<BOT>: ')[-1]
    else:
        response = full_text
    
    # Remove <END> token and any trailing special tokens
    response = response.replace('<END>', '').strip()
    
    return response

def test_model_responses(model, test_prompts=None):
    """Test the model with various prompts."""
    if test_prompts is None:
        # Test with different character prompts
        test_prompts = [
            "[ROMEO] What say you, Mercutio?",
            "[HAMLET] To be or not to be, that is the question.",
            "[JULIET] O Romeo, Romeo! Wherefore art thou Romeo?",
            "[MACBETH] Is this a dagger which I see before me?",
            "[LEAR] Blow, winds, and crack your cheeks!",
            "Tell me a story in the style of Shakespeare",
            "What thinkest thou of love?",
            "Speak to me of honor and betrayal",
        ]
    
    print("\n" + "="*60)
    print("TESTING SHAKESPEAREAN RESPONSES")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"You: {prompt}")
        
        response = generate_shakespearean_response(
            model, 
            prompt, 
            max_new_tokens=200
        )
        
        print(f"Bot: {response}")
        
        if i < len(test_prompts):
            print("-" * 40)

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize model
    model = LlamaModel(MASTER_CONFIG)
    model.to(device)
    
    # Try to load pre-trained weights
    pre_trained_path = "./models/llama_model.pth"
    if os.path.exists(pre_trained_path):
        print(f"Found pre-trained model at {pre_trained_path}")
        load_choice = input("Load pre-trained model? (y/n): ").lower()
        if load_choice == 'y':
            try:
                model.load_state_dict(torch.load(pre_trained_path))
                print("Model loaded successfully!")
                
                # Test the loaded model
                test_model_responses(model)
                
                # Ask if user wants to continue training
                continue_train = input("\nContinue training? (y/n): ").lower()
                if continue_train != 'y':
                    print("Exiting training. Use inference.py for chatting.")
                    exit(0)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Training from scratch...")
        else:
            print("Training from scratch...")
    else:
        print("No pre-trained model found. Training from scratch...")
    
    # Train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=MASTER_CONFIG['epochs']
    )
    
    print(f"\nStarting training for {MASTER_CONFIG['epochs']} epochs...")
    print(f"Batch size: {MASTER_CONFIG['batch_size']}")
    print(f"Context window: {MASTER_CONFIG['context_window']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train(model, optimizer, scheduler, print_logs=True)
    print("Training complete!")
    
    # Test the trained model
    test_model_responses(model)
    
    # Save the model
    print("\nSaving model...")
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), pre_trained_path)
    
    # Save vocabulary
    vocab_data = {
        "stoi": stoi,
        "itos": itos
    }
    with open("vocab.json", "w") as f:
        json.dump(vocab_data, f)
    print(f"Model saved to {pre_trained_path}")
    print("Vocabulary saved to vocab.json")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nTo chat with the Shakespearean bot, run:")
    print("  python inference.py")
    print("\nTo test specific prompts, run:")
    print("  python inference.py --test")
    print("  python inference.py --prompt \"[ROMEO] What say you?\"")