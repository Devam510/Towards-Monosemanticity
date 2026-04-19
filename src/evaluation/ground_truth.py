"""
Ground Truth Evaluation for Synthetic Models

Evaluates how well recovered SAE features match the planted true features.

Core metric: for each true feature direction d_true, find the SAE feature
direction d_sae with maximum cosine similarity. A feature is "recovered"
if max cosine sim ≥ threshold (default 0.9).

Also computes:
- Feature recovery rate (fraction of true features recovered)
- Mean max cosine similarity
- False positive rate (SAE features with no matching true feature)
- Cross-seed convergence (feature matching across different training runs)
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional


def cosine_similarity_matrix(
    dirs_a: torch.Tensor,  # [N, D]
    dirs_b: torch.Tensor   # [M, D]
) -> torch.Tensor:
    """
    Compute pairwise cosine similarities.
    Returns: [N, M] matrix where entry [i,j] = cos_sim(a_i, b_j)
    """
    a = F.normalize(dirs_a, dim=1)  # [N, D]
    b = F.normalize(dirs_b, dim=1)  # [M, D]
    return a @ b.T                   # [N, M]


def feature_recovery_rate(
    true_dirs: torch.Tensor,   # [C, D] — planted feature directions
    sae_dirs: torch.Tensor,    # [F, D] — recovered SAE feature directions
    threshold: float = 0.9
) -> dict:
    """
    Compute how many true features are recovered by the SAE.

    A true feature is "recovered" if at least one SAE feature has
    cosine similarity ≥ threshold with it.

    Returns dict with:
        recovery_rate: fraction of true features recovered (0 to 1)
        mean_max_cos: mean of max cosine similarities per true feature
        false_positive_rate: fraction of SAE features that don't match any true feature
        matched_pairs: list of (true_idx, sae_idx, cos_sim) for matched features
    """
    with torch.no_grad():
        cos_sim = cosine_similarity_matrix(true_dirs, sae_dirs)  # [C, F]

        # For each true feature, find the best matching SAE feature
        max_cos_per_true, best_sae_idx = cos_sim.max(dim=1)  # [C]

        # Recovery: true features with max cosine sim >= threshold
        recovered = (max_cos_per_true >= threshold)
        recovery_rate = recovered.float().mean().item()
        mean_max_cos = max_cos_per_true.mean().item()

        # False positives: SAE features that don't match any true feature
        max_cos_per_sae, _ = cos_sim.max(dim=0)  # [F]
        false_positives = (max_cos_per_sae < threshold)
        false_positive_rate = false_positives.float().mean().item()

        # Build matched pairs list
        matched_pairs = [
            (i, best_sae_idx[i].item(), max_cos_per_true[i].item())
            for i in range(len(true_dirs))
            if recovered[i].item()
        ]

    return {
        "recovery_rate": recovery_rate,
        "mean_max_cos": mean_max_cos,
        "false_positive_rate": false_positive_rate,
        "n_recovered": recovered.sum().item(),
        "n_true": len(true_dirs),
        "n_sae": len(sae_dirs),
        "matched_pairs": matched_pairs,
    }


def cross_seed_convergence(
    sae_dirs_list: list[torch.Tensor],  # List of [F, D] from different seeds
    threshold: float = 0.9
) -> dict:
    """
    Measure how consistently the same features emerge across training seeds.

    For each pair of runs, uses Hungarian matching to optimally align features,
    then measures how many matched pairs have cosine sim >= threshold.

    Returns:
        mean_match_cos: mean cosine similarity of matched feature pairs
        convergence_rate: fraction of features with a close match across seeds
        pairwise_results: list of per-pair match results
    """
    n_runs = len(sae_dirs_list)
    if n_runs < 2:
        raise ValueError("Need at least 2 runs to measure convergence")

    pairwise = []
    all_match_cos = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            dirs_a = sae_dirs_list[i]  # [F_a, D]
            dirs_b = sae_dirs_list[j]  # [F_b, D]

            cos_sim = cosine_similarity_matrix(dirs_a, dirs_b).detach().cpu().numpy()

            # Hungarian algorithm: maximize total cosine similarity of matched pairs
            # scipy expects a cost matrix (minimize), so negate
            F_a, F_b = cos_sim.shape
            n_match = min(F_a, F_b)
            row_ind, col_ind = linear_sum_assignment(-cos_sim)

            matched_cos = cos_sim[row_ind, col_ind]  # [n_match]
            good_matches = (matched_cos >= threshold).sum()

            pairwise.append({
                "run_i": i,
                "run_j": j,
                "mean_match_cos": float(matched_cos.mean()),
                "convergence_rate": float(good_matches / n_match),
                "n_matched": n_match,
            })
            all_match_cos.extend(matched_cos.tolist())

    return {
        "mean_match_cos": float(np.mean(all_match_cos)),
        "convergence_rate": float(np.mean([p["convergence_rate"] for p in pairwise])),
        "pairwise_results": pairwise,
    }


def explained_variance(
    x: torch.Tensor,      # [batch, D] original activations
    x_hat: torch.Tensor   # [batch, D] reconstructed activations
) -> float:
    """Fraction of variance explained by reconstruction."""
    residual_var = ((x - x_hat) ** 2).mean().item()
    total_var = x.var().item()
    return max(0.0, 1.0 - residual_var / (total_var + 1e-8))
