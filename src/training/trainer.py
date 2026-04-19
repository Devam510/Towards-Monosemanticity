"""
Generic SAE Trainer

Handles the training loop for any SAE variant. Core responsibilities:
- Optimizer setup
- Training loop with logging
- Decoder normalization after each step (critical for interpretability)
- Dead feature tracking and resurrection (ghost grads lite)
- Saving checkpoints

Design: trainer is decoupled from model — works with any nn.Module
that exposes a .forward() returning (x_hat, z, loss).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Callable
from tqdm import tqdm


@dataclass
class TrainerConfig:
    n_steps: int = 20_000
    batch_size: int = 2048
    lr: float = 2e-4
    lr_warmup_steps: int = 1000
    log_every: int = 200
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Dead feature detection
    dead_feature_window: int = 1000  # Steps before calling a feature dead
    dead_feature_threshold: float = 1e-8


class SAETrainer:
    """
    Generic trainer for SparseAutoencoder models.
    
    Usage:
        trainer = SAETrainer(sae, config)
        trainer.train(data_iter)
    """

    def __init__(self, model: nn.Module, config: TrainerConfig):
        self.model = model.to(config.device)
        self.config = config

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, betas=(0.9, 0.999)
        )

        # LR warmup scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.lr_warmup_steps
        )

        # Dead feature tracking
        self.feature_activation_counts = None
        self.steps_since_activation = None

        self.log: list[dict] = []

    def _warmup_lr(self, step: int):
        if step < self.config.lr_warmup_steps:
            self.scheduler.step()

    def _track_dead_features(self, z: torch.Tensor):
        """Track which features haven't activated recently."""
        n_features = z.shape[1]
        if self.steps_since_activation is None:
            self.steps_since_activation = torch.zeros(n_features, device=z.device)

        active = (z.abs() > self.config.dead_feature_threshold).any(dim=0).float()
        # Increment counter for inactive features, reset for active
        self.steps_since_activation = (self.steps_since_activation + 1) * (1 - active)

    def _n_dead_features(self) -> int:
        if self.steps_since_activation is None:
            return 0
        return (self.steps_since_activation > self.config.dead_feature_window).sum().item()

    def _normalize_decoder(self):
        """Keep decoder columns unit norm after each optimizer step."""
        if hasattr(self.model, '_normalize_decoder'):
            self.model._normalize_decoder()

    def train(
        self,
        data_iter,  # Iterator yielding [batch, d_model] tensors
        callback: Optional[Callable] = None
    ) -> list[dict]:
        """
        Train the SAE.

        Args:
            data_iter: iterator or generator of activation batches
            callback: optional function called with (step, log_entry) for logging

        Returns:
            log: list of dicts with training metrics
        """
        self.model.train()
        cfg = self.config

        pbar = tqdm(range(cfg.n_steps), desc="Training SAE")
        data_iter_iter = iter(data_iter)

        for step in pbar:
            try:
                batch = next(data_iter_iter).to(cfg.device)
            except StopIteration:
                # Restart iterator
                data_iter_iter = iter(data_iter)
                batch = next(data_iter_iter).to(cfg.device)

            self.optimizer.zero_grad()

            x_hat, z, loss = self.model(batch)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self._warmup_lr(step)
            self._normalize_decoder()
            self._track_dead_features(z.detach())

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    recon = ((batch - x_hat) ** 2).mean().item()
                    l0 = (z.abs() > 1e-8).float().sum(dim=1).mean().item()
                    n_dead = self._n_dead_features()
                    sparsity = z.abs().mean().item()

                entry = {
                    "step": step,
                    "loss": loss.item(),
                    "recon_mse": recon,
                    "explained_var": 1 - recon / batch.var().item(),
                    "l0": l0,
                    "n_dead": n_dead,
                    "sparsity_l1": sparsity,
                }
                self.log.append(entry)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    recon=f"{recon:.4f}",
                    l0=f"{l0:.1f}",
                    dead=n_dead
                )

                if callback:
                    callback(step, entry)

        self.model.eval()
        return self.log
