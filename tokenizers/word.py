"""
Word-Level Tokenizer
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from typing import List, Dict, Iterable, Optional

# Configuration
@dataclass
class TokenizerConfig:
    lowercase: bool = True
    min_frequency: int = 1
    max_vocab_size: Optional[int] = None
    keep_punctuation: bool = False
    
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"

# Tokenizer
class WordTokenizer:
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        self.is_fitted: bool = False
    
    # Internal text normalization
    def _normalize(self, text: str) -> str:
        if self.config.lowercase:
            text = text.lower()
        
        text = text.strip()
        
        if self.config.keep_punctuation:
            text = re.sub(r"([.,!?;:()])", r"\1", text)
        else:
            text = re.sub(r"[^\w\s]", "", text)
        
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        text = self._normalize(text)
        if not text:
            return []
        
        return text.split(" ")

    def fit(self, corpus: Iterable[str]) -> None:
        counter = Counter()
        
        for text in corpus:
            tokens = self._tokenize(text)
            counter.update(tokens)
        
        # Apply minimum frequency
        items = [
            (token, freq)
            for token, freq in counter.items()
            if freq >= self.config.min_frequency
        ]
        
        # Sort:
        # 1. Higher frequency first
        # 2. Alphabetically for determination
        
        items.sort(key=lambda x: (-x[1], x[0]))
        
        # Apply vocab cap
        if self.config.max_vocab_size is not None:
            items = items[: self.config.max_vocab_size]
        
        vocab_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
        ]
        
        vocab_tokens.extend(token for token, _ in items)
        
        self.token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        self.is_fitted = True
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        
        self._ensure_fitted()
        
        tokens = self._tokenize(text)
        
        ids = []
        
        if add_special_tokens:
            ids.append(self.token_to_id[self.config.bos_token])
        
        unk_id = self.token_to_id[self.config.unk_token]
        
        for token in tokens:
            ids.append(self.token_to_id.get(token, unk_id))
        
        if add_special_tokens:
            ids.append(self.token_to_id[self.config.eos_token])
        
        return ids

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
    ) -> List[List[int]]:
        
        batch = [self.encode(t) for t in texts]
        
        if not padding:
            return batch
        
        pad_id = self.token_to_id[self.config.pad_token]
        max_len = max(len(seq) for seq in batch)
        
        padded = []
        for seq in batch:
            padded.append(seq + [pad_id] * (max_len - len(seq)))
        
        return padded
    
    
