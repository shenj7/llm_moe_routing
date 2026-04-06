"""
Dataset loading for LLaDA progressive distillation.

Streams text from FineWeb — the closest publicly available proxy to LLaDA's
training corpus (which is a private 2.3 T-token mixture of web text, code,
math, and multilingual content).

The dataset returns *clean* token sequences (x_0).  Masking (the forward
diffusion process) is applied dynamically inside the training loop so that
each epoch sees a different noise realisation.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaskedDiffusionDataset(IterableDataset):
    """
    Streaming dataset that tokenises text and yields non-overlapping chunks
    of fixed length as clean token tensors ready for masked diffusion training.

    Args:
        tokenizer:       HuggingFace tokenizer (loaded from teacher).
        dataset_name:    HuggingFace dataset repo (default: FineWeb).
        dataset_config:  Dataset config / subset name.
        text_column:     Column that contains the raw text.
        max_seq_length:  Length of each output token chunk.
        split:           Dataset split to use.
        seed:            Shuffle seed for the streaming buffer.
        streaming:       Whether to use HF streaming mode.
        buffer_size:     Shuffle buffer size (streaming only).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str = "HuggingFaceFW/fineweb",
        dataset_config: str = "sample-10BT",
        text_column: str = "text",
        max_seq_length: int = 128,
        split: str = "train",
        seed: int = 42,
        streaming: bool = True,
        buffer_size: int = 10_000,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.max_seq_length = max_seq_length
        self.split = split
        self.seed = seed
        self.streaming = streaming
        self.buffer_size = buffer_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_hf_dataset(self):
        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            streaming=self.streaming,
        )
        if self.streaming:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)
        return ds

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[torch.Tensor]:
        dataset = self._build_hf_dataset()
        buffer: list[int] = []

        for example in dataset:
            text: str = example[self.text_column]

            token_ids: list[int] = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
            buffer.extend(token_ids)

            # Yield non-overlapping fixed-length chunks.
            while len(buffer) >= self.max_seq_length:
                chunk = buffer[: self.max_seq_length]
                buffer = buffer[self.max_seq_length :]
                yield torch.tensor(chunk, dtype=torch.long)


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack a list of 1-D token tensors into a 2-D batch tensor [B, L]."""
    return torch.stack(batch, dim=0)
