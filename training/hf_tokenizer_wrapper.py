"""
HuggingFace Tokenizer Wrapper for Scrappy LLM.
Bridges HuggingFace tokenizers library to the Scrappy tokenizer interface.
"""

from tokenizers import Tokenizer as HFTokenizer


class HFTokenizerWrapper:
    """Wrapper for HuggingFace tokenizers to match Scrappy interface"""

    def __init__(self, tokenizer_path: str):
        """Initialize wrapper from HuggingFace tokenizer file.

        Args:
            tokenizer_path: Path to HuggingFace tokenizer JSON file
        """
        self.hf_tokenizer = HFTokenizer.from_file(tokenizer_path)
        self.vocab_size = self.hf_tokenizer.get_vocab_size()

        # Map special tokens
        self.pad_id = self.hf_tokenizer.token_to_id("<PAD>")
        self.unk_id = self.hf_tokenizer.token_to_id("<UNK>")
        self.bos_id = self.hf_tokenizer.token_to_id("<BOS>")
        self.eos_id = self.hf_tokenizer.token_to_id("<EOS>")
        self.eot_id = self.eos_id  # Compatibility with existing code

        print(f"HFTokenizerWrapper loaded: vocab_size={self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input string

        Returns:
            List of token IDs
        """
        encoding = self.hf_tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens from output

        Returns:
            Decoded string
        """
        decoded = self.hf_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        # Clean up BPE tokenizer artifacts:
        # 1. Remove space-marker character if present (Ġ)
        decoded = decoded.replace('Ġ', ' ')

        # 2. Fix specific BPE spacing issues
        import re

        # Pattern 1: Fix contractions like "didn ' t" -> "didn't"
        decoded = re.sub(r" ' (t|s|m|d|ll|ve|re)\b", r"'\1", decoded)

        # Pattern 2: Collapse multiple consecutive spaces
        decoded = re.sub(r' +', ' ', decoded)

        return decoded.strip()

    def save(self, filepath: str):
        """Save tokenizer (delegates to HuggingFace).

        Args:
            filepath: Path to save tokenizer JSON
        """
        self.hf_tokenizer.save(filepath)
        print(f"Saved HF tokenizer to {filepath}")
