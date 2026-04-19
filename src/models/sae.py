"""
Sparse Autoencoder with TopK Activation

Standard SAE baseline. Uses TopK activation to enforce EXACT sparsity
(exactly k features active per input).

Why TopK instead of L1:
- L1 requires careful tuning of the coefficient relative to data scale
- When reconstruction error -> 0, L1 is too weak to force sparsity
- TopK directly controls L0: exactly k features are always active
- Much more stable in practice for synthetic experiments where k is known

For real model experiments, we use L1 (since k is unknown), but for
synthetic validation we always use TopK since k is known by construction.

Architecture:
    pre = W_enc @ (x - b_pre) + b_enc    # Encoder pre-activation
    z = TopK(pre, k)                       # Keep only k largest
    x_hat = W_dec @ z + b_pre             # Decode

Loss:
    L = MSE(x, x_hat) + lambda * ||z||_1  # Reconstruction + sparsity regularization
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SAEConfig:
    d_model: int = 256          # Input activation dimension
    n_features: int = 1024      # Dictionary size (overcomplete)
    l1_coeff: float = 1e-3      # Sparsity regularization weight (used in L1 mode)
    k: int = 0                  # TopK — if 0, use L1 mode instead
    lr: float = 2e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TopKActivation(nn.Module):
    """
    TopK sparse activation: keeps the k largest activations, zeros the rest.
    Uses straight-through estimator for the k-selection (gradient flows through
    the selected activations as if they were never masked).
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n_features]
        Returns: [batch, n_features] with all but top-k zeroed out
        """
        if self.k <= 0:
            return torch.relu(x)

        # Get top-k values and indices
        topk_vals, topk_idx = torch.topk(x, self.k, dim=1)  # [batch, k]

        # Build sparse output
        z = torch.zeros_like(x)
        z.scatter_(1, topk_idx, torch.relu(topk_vals))  # Only keep positive topk
        return z


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with configurable sparsity mechanism:
    - TopK mode (k > 0): exactly k features active — USE FOR SYNTHETIC
    - L1 mode (k == 0): soft sparsity via L1 penalty — USE FOR REAL MODELS
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)

        D, F = config.d_model, config.n_features

        # Pre-encoder bias
        self.b_pre = nn.Parameter(torch.zeros(D))

        # Encoder
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(F, D)))
        self.b_enc = nn.Parameter(torch.zeros(F))

        # Activation — TopK if k specified, else learned threshold ReLU
        if config.k > 0:
            self.activation = TopKActivation(config.k)
        else:
            # Learnable threshold: z = ReLU(pre - theta)
            self.activation = _LearnedThresholdReLU(F)

        # Decoder (init as transpose)
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone())

        self._normalize_decoder()

    def _normalize_decoder(self):
        """Unit-norm decoder columns — critical to prevent trivial large-magnitude solutions."""
        with torch.no_grad():
            self.W_dec.data = nn.functional.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, D] -> z: [batch, F]"""
        pre = (x - self.b_pre) @ self.W_enc.T + self.b_enc
        return self.activation(pre)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [batch, F] -> x_hat: [batch, D]"""
        return z @ self.W_dec.T + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z, loss)"""
        z = self.encode(x)
        x_hat = self.decode(z)

        recon_loss = ((x - x_hat) ** 2).mean()

        # Sparsity: L1 in L1 mode, or soft L1 in TopK mode (mild regularization)
        sparsity_loss = self.config.l1_coeff * z.abs().mean()
        loss = recon_loss + sparsity_loss

        return x_hat, z, loss

    @property
    def decoder_directions(self) -> torch.Tensor:
        """Normalized decoder columns. [D, F]"""
        return nn.functional.normalize(self.W_dec, dim=0)

    def feature_directions(self) -> torch.Tensor:
        """Feature directions as [F, D]."""
        return self.decoder_directions.T


class _LearnedThresholdReLU(nn.Module):
    """z = ReLU(x - theta) with log-parameterized positive threshold."""

    def __init__(self, n_features: int, init_threshold: float = 0.1):
        super().__init__()
        self.log_threshold = nn.Parameter(
            torch.full((n_features,), fill_value=float(np.log(init_threshold)))
        )

    @property
    def threshold(self) -> torch.Tensor:
        return torch.exp(self.log_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x - self.threshold.to(x.device), min=0.0)


# Backward-compatible alias
JumpReLU = _LearnedThresholdReLU
