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

        # 2. Fix spacing issues - remove spaces within words
        # Common pattern: single letters with spaces should be joined
        import re

        # Pattern 1: Fix single character fragments like "r ome o" -> "romeo"
        # Look for pattern of letter-space-letter where both are lowercase
        decoded = re.sub(r'\b([a-z]) ([a-z]{1,3})\b', r'\1\2', decoded)
        decoded = re.sub(r'\b([a-z]) ([a-z]{1,3}) ([a-z])\b', r'\1\2\3', decoded)

        # Pattern 2: Fix names like "Ca i us" -> "Caius"
        decoded = re.sub(r'\b([A-Z][a-z]?) ([a-z]) ([a-z]+)\b', r'\1\2\3', decoded)
        decoded = re.sub(r'\b([A-Z][a-z]+) ([a-z]) ([a-z]+)\b', r'\1\2\3', decoded)

        # Pattern 3: Collapse multiple spaces
        decoded = re.sub(r' +', ' ', decoded)

        # Pattern 4: Fix contractions like "didn ' t" -> "didn't"
        decoded = re.sub(r" ' (t|s|m|d|ll|ve|re)\b", r"'\1", decoded)

        return decoded.strip()

    def save(self, filepath: str):
        """Save tokenizer (delegates to HuggingFace).

        Args:
            filepath: Path to save tokenizer JSON
        """
        self.hf_tokenizer.save(filepath)
        print(f"Saved HF tokenizer to {filepath}")
