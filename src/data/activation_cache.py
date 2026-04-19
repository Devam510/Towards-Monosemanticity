"""
Activation Cache — Memory-Efficient Disk-Based Storage

Caches model activations to disk so SAE training doesn't require
the base model in GPU memory. Critical for 4GB VRAM constraint.

Strategy:
1. Run base model once, save activations in chunks
2. SAE training loads random chunks from disk
3. Base model can be unloaded after caching

Format: torch .pt files, one per chunk, stored in cache_dir
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Generator


class ActivationCache:
    """
    Disk-backed store for model activations at specific layers.
    
    Saves to: cache_dir/{layer_name}/{chunk_idx}.pt
    Each chunk is a tensor of shape [chunk_size, d_model]
    """

    def __init__(self, cache_dir: str, layer_names: list[str], chunk_size: int = 2048):
        self.cache_dir = Path(cache_dir)
        self.layer_names = layer_names
        self.chunk_size = chunk_size
        self.chunk_counts: dict[str, int] = {}  # layer -> number of chunks

        # Create directories and read existing chunks
        for name in layer_names:
            layer_dir = self.cache_dir / name
            layer_dir.mkdir(parents=True, exist_ok=True)
            
            # Auto-detect existing chunks if any
            existing_chunks = list(layer_dir.glob("*.pt"))
            if len(existing_chunks) > 0:
                self.chunk_counts[name] = len(existing_chunks)

    def save_batch(self, layer_name: str, activations: torch.Tensor):
        """
        Save a batch of activations to disk.
        activations: [batch, d_model]
        """
        assert layer_name in self.layer_names

        count = self.chunk_counts.get(layer_name, 0)
        path = self.cache_dir / layer_name / f"{count:05d}.pt"
        torch.save(activations.cpu().half(), path)  # Save as fp16 to save disk
        self.chunk_counts[layer_name] = count + 1

    def load_chunk(self, layer_name: str, chunk_idx: int) -> torch.Tensor:
        """Load a specific chunk. Returns [chunk_size, d_model] float32."""
        path = self.cache_dir / layer_name / f"{chunk_idx:05d}.pt"
        return torch.load(path, weights_only=True).float()

    def iter_random_batches(
        self,
        layer_name: str,
        batch_size: int,
        n_steps: int,
        device: str = "cpu"
    ) -> Generator[torch.Tensor, None, None]:
        """
        Yield random batches from cached activations.
        Used as the data loader for SAE training.
        """
        n_chunks = self.chunk_counts[layer_name]
        assert n_chunks > 0, f"No chunks cached for layer {layer_name}"

        for _ in range(n_steps):
            # Pick a random chunk
            chunk_idx = np.random.randint(0, n_chunks)
            chunk = self.load_chunk(layer_name, chunk_idx).to(device)

            # Sample batch_size rows from chunk
            if len(chunk) >= batch_size:
                idx = torch.randperm(len(chunk))[:batch_size]
                yield chunk[idx]
            else:
                yield chunk

    def n_chunks(self, layer_name: str) -> int:
        return self.chunk_counts.get(layer_name, 0)
        
    def get_random_batch_all(self, layer_names: list[str], batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        """
        Load a batch of identical tokens across multiple layers.
        This is absolutely critical for inter-layer consistency training.
        """
        n_chunks = self.chunk_counts[layer_names[0]]
        chunk_idx = np.random.randint(0, n_chunks)
        
        chunks = {name: self.load_chunk(name, chunk_idx) for name in layer_names}
        
        chunk_len = len(chunks[layer_names[0]])
        if chunk_len >= batch_size:
            idx = torch.randperm(chunk_len)[:batch_size]
            return {name: chunks[name][idx].to(device) for name in layer_names}
        return {name: chunks[name].to(device) for name in layer_names}

    def total_samples(self, layer_name: str) -> int:
        return self.chunk_counts.get(layer_name, 0) * self.chunk_size
