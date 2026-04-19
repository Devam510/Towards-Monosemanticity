"""
CCJFR Trainer

Handles the training loop for the CCJFR model.
Optimizes all SAEs jointly with the specified losses.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Callable
from tqdm import tqdm


@dataclass
class CCJFRTrainerConfig:
    n_steps: int = 15_000
    batch_size: int = 2048
    lr: float = 2e-4
    lr_warmup_steps: int = 1000
    log_every: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CCJFRTrainer:
    def __init__(self, model: nn.Module, config: CCJFRTrainerConfig):
        self.model = model.to(config.device)
        self.config = config

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.lr_warmup_steps
        )
        self.log: list[dict] = []

    def _warmup_lr(self, step: int):
        if step < self.config.lr_warmup_steps:
            self.scheduler.step()

    def train(
        self,
        data_iter,  # Iterator yielding (activations_list, actual_logits)
        compute_fns: Optional[list[Callable]] = None,
        callback: Optional[Callable] = None
    ) -> list[dict]:
        self.model.train()
        cfg = self.config

        pbar = tqdm(range(cfg.n_steps), desc="Training CCJFR")
        data_iter_iter = iter(data_iter)

        for step in pbar:
            try:
                batch_data = next(data_iter_iter)
            except StopIteration:
                data_iter_iter = iter(data_iter)
                batch_data = next(data_iter_iter)

            activations = [h.to(cfg.device) for h in batch_data["activations"]]
            actual_logits = batch_data.get("logits", None)
            if actual_logits is not None:
                actual_logits = actual_logits.to(cfg.device)

            self.optimizer.zero_grad()

            total_loss, metrics = self.model(
                activations=activations,
                compute_fns=compute_fns,
                actual_logits=actual_logits
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self._warmup_lr(step)
            self.model.normalize_all_decoders()

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    # For logging sparsity logic
                    l0s = []
                    for z in metrics["feature_acts"]:
                        l0 = (z.abs() > 1e-8).float().sum(dim=1).mean().item()
                        l0s.append(l0)

                metrics_entry = {
                    "step": step,
                    "loss": total_loss.item(),
                    "recon_loss": metrics["recon_loss"],
                    "consist_loss": metrics["consist_loss"],
                    "embed_anchor_loss": metrics["embed_anchor_loss"],
                    "unembed_anchor_loss": metrics["unembed_anchor_loss"],
                    "consist_coeff": metrics["consist_coeff"],
                    "l0_mean": sum(l0s) / len(l0s),
                    "l0_by_layer": l0s,
                }
                self.log.append(metrics_entry)

                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    recon=f"{metrics['recon_loss']:.4f}",
                    consist=f"{metrics['consist_loss']:.4f}",
                    l0=f"{l0s[0]:.1f}"
                )

                if callback:
                    callback(step, metrics_entry)

        self.model.eval()
        return self.log
