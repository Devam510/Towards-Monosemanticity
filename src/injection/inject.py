"""
Feature Injection Test

Creates synthetic ground truth INSIDE a real (or synthetic) model by injecting
a known rank-1 perturbation into a weight matrix, then testing whether the
SAE recovers the injected feature direction.

Why this matters:
- We can't get ground truth from a real model
- BUT we can ADD a known feature to the model's weights
- If the SAE finds it, we have direct evidence it can recover real features
- Varying the injection strength α tests the detection limit

Usage:
    injector = FeatureInjector(model, layer=2, matrix_name='W_out')
    model_perturbed = injector.inject(u_dir, v_dir, alpha=0.5)
    # ... train SAE on model_perturbed ...
    result = injector.verify_recovery(sae.feature_directions(), threshold=0.9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional


class FeatureInjector:
    """
    Injects a rank-1 feature into a model weight matrix.

    The injection: W' = W + α * v @ u.T
    - u: input direction (D_in,) — which input directions activate this feature
    - v: output direction (D_out,) — how the feature writes to the next layer
    - α: injection strength

    A larger α makes the injected feature stronger and easier to recover.
    """

    def __init__(self):
        self.injected_features: list[dict] = []  # Track all injections

    def inject_into_tensor(
        self,
        W: torch.Tensor,   # [D_out, D_in] weight matrix
        u: torch.Tensor,   # [D_in] input direction (normalized)
        v: torch.Tensor,   # [D_out] output direction (normalized)
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Return W' = W + alpha * v.unsqueeze(1) @ u.unsqueeze(0)
        """
        u = F.normalize(u, dim=0)
        v = F.normalize(v, dim=0)
        perturbation = alpha * v.unsqueeze(1) @ u.unsqueeze(0)  # [D_out, D_in]
        return W + perturbation

    def record_injection(self, u: torch.Tensor, v: torch.Tensor, alpha: float, layer: int):
        """Record a feature injection for later verification."""
        self.injected_features.append({
            "u": u.detach().cpu(),   # Input direction
            "v": v.detach().cpu(),   # Output direction
            "alpha": alpha,
            "layer": layer,
        })

    def verify_recovery(
        self,
        sae_dirs: torch.Tensor,  # [F, D] SAE feature directions
        injection_idx: int = -1,  # Which injection to verify (-1 = last)
        threshold: float = 0.9,
        direction: str = "output"  # "input" or "output" direction to check
    ) -> dict:
        """
        Check if the injected feature was recovered by the SAE.

        Args:
            sae_dirs: [F, D] SAE decoder directions
            injection_idx: which injection to check
            threshold: cosine similarity threshold for "recovered"
            direction: check input direction u or output direction v

        Returns dict with:
            recovered: bool
            max_cos: highest cosine similarity with any SAE feature
            best_feature_idx: which SAE feature matched best
        """
        record = self.injected_features[injection_idx]
        target_dir = record["v"] if direction == "output" else record["u"]
        target_dir = F.normalize(target_dir, dim=0)  # [D]

        sae_dirs_norm = F.normalize(sae_dirs.cpu(), dim=1)  # [F, D]
        cos_sims = (sae_dirs_norm @ target_dir).abs()  # [F] — use abs for sign flip

        max_cos = cos_sims.max().item()
        best_idx = cos_sims.argmax().item()

        return {
            "recovered": max_cos >= threshold,
            "max_cos": max_cos,
            "best_feature_idx": best_idx,
            "alpha": record["alpha"],
            "layer": record["layer"],
            "threshold": threshold,
        }


def generate_injection_directions(d_model: int, seed: int = 0) -> tuple:
    """
    Generate a random pair of (u, v) injection directions.
    These are orthogonal to each other for clean injection.

    Returns: (u, v) each [d_model]
    """
    torch.manual_seed(seed)
    u = F.normalize(torch.randn(d_model), dim=0)
    # Make v orthogonal to u
    v_raw = torch.randn(d_model)
    v_raw = v_raw - (v_raw @ u) * u  # Project out u component
    v = F.normalize(v_raw, dim=0)
    return u, v
