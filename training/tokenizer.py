"""
BPE Tokenizer wrapper for Scrappy LLM.
Adapted from pirate_llm-main tokenization pipeline.
"""

import json
import numpy as np
import torch
from pathlib import Path


class Tokenizer:
    @staticmethod
    def auto_load(tokenizer_path: str):
        """Auto-detect tokenizer format and load appropriately.

        Args:
            tokenizer_path: Path to tokenizer JSON file

        Returns:
            Tokenizer instance (either Tokenizer, HFTokenizerWrapper, or CharTokenizer)
        """
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # HuggingFace format has "model" key with "type": "BPE"
        if "model" in data and data.get("model", {}).get("type") == "BPE":
            from .hf_tokenizer_wrapper import HFTokenizerWrapper
            return HFTokenizerWrapper(tokenizer_path)
        # CharTokenizer format has "type": "CharTokenizer"
        elif "type" in data and data["type"] == "CharTokenizer":
            # Load CharTokenizer from saved format
            tokenizer = CharTokenizer.__new__(CharTokenizer)
            tokenizer.vocab_size = data['vocab_size']
            tokenizer.stoi = data['stoi']
            tokenizer.itos = {int(k): v for k, v in data['itos'].items()}
            return tokenizer
        else:
            # Original simple JSON format
            return Tokenizer(tokenizer_path)

    """Simple BPE tokenizer using JSON vocabulary."""
    
    def __init__(self, vocab_path: str, special_tokens: dict = None):
        """Initialize tokenizer from JSON vocab file.
        
        Args:
            vocab_path: Path to JSON file with vocab mapping
            special_tokens: Optional special token mappings (e.g., {"EOT": 1})
        """
        self.vocab_path = Path(vocab_path)
        self.special_tokens = special_tokens or {}

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data.get('vocab', {})
        self.special_tokens_inv = {v: k for k, v in self.special_tokens.items()}
        
        # Build id-to-token mapping
        self.id_to_token = {}
        for token_id, token_str in self.vocab.items():
            token_id_int = int(token_id)
            self.id_to_token[token_id_int] = token_str
        
        # Add special tokens
        for token, token_id in self.special_tokens.items():
            self.id_to_token[token_id] = token
        
        self.vocab_size = len(self.vocab) + len(self.special_tokens)
        self.eot_id = self.special_tokens.get('EOT', self.vocab_size - 1)
        
        print(f"Tokenizer loaded: {self.vocab_size} tokens, EOT={self.eot_id}")
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input string
            
        Returns:
            List of token IDs
        """
        ids = []
        for token in text.split():
            if token in self.vocab:
                ids.append(int(self.vocab[token]))
            elif token in self.special_tokens_inv:
                ids.append(self.special_tokens_inv[token])
            else:
                # Fallback: character-level encoding
                for char in token:
                    char_id = ord(char)
                    if char_id < self.vocab_size:
                        ids.append(char_id)
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded string
        """
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                tokens.append(self.id_to_token[idx])
        
        return ' '.join(tokens)
    
    @classmethod
    def from_file(cls, filepath: str) -> "Tokenizer":
        """Load tokenizer from file."""
        return cls(filepath)
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
            'eot_id': self.eot_id
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved tokenizer to {filepath}")
    
    def load_vocab(self, vocab: dict, char_vocab: dict = None):
        """Load character-based vocabulary for training from text data.
        
        Args:
            vocab: Character vocabulary dict from text analysis
            char_vocab: Optional character vocab dict
        """
        if char_vocab:
            # Start from character vocabulary
            self.vocab = {str(i): ch for i, ch in char_vocab.items()}
            self.vocab.update(vocab)
        else:
            # Build from set of characters
            chars = sorted(list(vocab))
            self.vocab = {str(i): ch for i, ch in enumerate(chars)}
        
        self.vocab_size = len(self.vocab)
        self.id_to_token = {int(k): v for k, v in self.vocab.items()}
        self.id_to_token.update({v: i for i, v in self.special_tokens.items()})


class CharTokenizer:
    """Simple character-level tokenizer (fallback for training)."""
    
    def __init__(self, text: str, special_tokens: dict = None):
        """Initialize with text corpus.
        
        Args:
            text: Text corpus to build vocab from
            special_tokens: Optional special token mappings
        """
        self.special_tokens = special_tokens or {}
        chars = sorted(list(set(text)))
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars) + len(self.special_tokens)
        
        if special_tokens:
            for token, idx in special_tokens.items():
                self.stoi[token] = idx
                self.itos[idx] = token
        
        print(f"Char tokenizer: {self.vocab_size} unique chars")
    
    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]
    
    def decode(self, ids: list[int]) -> str:
        return ''.join([self.itos[i] for i in ids])
    
    def save(self, filepath: str, vocab_data: dict = None):
        """Save tokenizer as JSON compatible with BPE format."""
        vocab = {str(i): ch for i, ch in self.itos.items() if i < self.vocab_size}
        data = {
            'vocab': vocab,
            'vocab_size': self.vocab_size,
            'eot_id': len(vocab) - 1
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
