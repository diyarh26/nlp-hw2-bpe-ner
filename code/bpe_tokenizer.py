"""
Your BPE tokenizer goes here.

Implement a Byte Pair Encoding tokenizer that subclasses BaseTokenizer.
This is the single module both train_tokenizer.py and generate_tokenizers.py
import, so keep the class here (or update their imports if you rename it).

Requirements (see the assignment spec):
  * True BPE: start from characters and merge upward.
  * Allow tokens up to the bigram level (two adjacent words -> one token) and
    produce at least one bigram.
  * Set the `space_token` attribute (it is set as None; the NER pipeline reads
    it and the submission check rejects a None value).
  * Implement train(), encode() and decode().
  * Only the provided data may be used for training.
"""

from typing import List

from base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        # Required by the NER pipeline for token<->word alignment.
        self.space_token = None

    def train(self, texts: List[str]) -> None:
        """Train the BPE tokenizer on a list of texts."""
        raise NotImplementedError("Implement BPETokenizer.train()")

    def encode(self, text: str) -> List[int]:
        """Convert a text string into a list of token IDs."""
        raise NotImplementedError("Implement BPETokenizer.encode()")

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back into a text string."""
        raise NotImplementedError("Implement BPETokenizer.decode()")
