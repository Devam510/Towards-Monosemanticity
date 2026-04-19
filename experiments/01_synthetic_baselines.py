"""
Experiment 01: Synthetic Baseline Experiments
Phase 1 — Step 1.3

Trains standard SAEs on synthetic data (no CCJFR constraints).
Measures:
- Feature recovery rate vs planted ground truth
- Explained variance and L0 sparsity

This establishes the BASELINE we need to beat.

Expected result: ≥ 60% recovery rate (Phase 1 gate criterion)

Run: python experiments/01_synthetic_baselines.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from src.models.synthetic_transformer import SyntheticTransformer, SyntheticConfig, get_true_features_at_layer
from src.models.sae import SparseAutoencoder, SAEConfig
from src.training.trainer import SAETrainer, TrainerConfig
from src.evaluation.ground_truth import feature_recovery_rate, explained_variance, cross_seed_convergence


def run_synthetic_baseline(
    layer: int = 0,
    n_features_sae: int = 512,
    seed: int = 42,
    n_steps: int = 10_000,
    batch_size: int = 1024,
    device: str = None
) -> dict:
    """Train one standard SAE on one layer of the synthetic model."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Standard SAE Baseline | Layer {layer} | Seed {seed}")
    print(f"{'='*60}")

    # --- Build synthetic model ---
    syn_config = SyntheticConfig(
        n_features=100,
        d_model=256,
        n_layers=3,
        sparsity_k=5,
        n_samples=50_000,
        seed=0  # Fixed model seed for reproducibility
    )
    model = SyntheticTransformer(syn_config)
    print(f"Synthetic model: {syn_config.n_features} features, {syn_config.n_layers} layers, d={syn_config.d_model}")

    # --- Generate data ---
    print("Generating synthetic data...")
    z_true, activations = model.generate_data(n_samples=50_000, seed=seed)
    h_layer = activations[layer]  # [50000, 256]
    print(f"Data shape at layer {layer}: {h_layer.shape}")

    # --- Train SAE ---
    sae_config = SAEConfig(
        d_model=syn_config.d_model,
        n_features=n_features_sae,
        k=syn_config.sparsity_k,   # TopK: force exact sparsity matching ground truth k
        l1_coeff=1e-4,             # Mild L1 on top of TopK (optional regularization)
        lr=2e-4,
        seed=seed,
        device=device
    )
    sae = SparseAutoencoder(sae_config)

    trainer_config = TrainerConfig(
        n_steps=n_steps,
        batch_size=batch_size,
        lr=2e-4,
        log_every=500,
        device=device
    )
    trainer = SAETrainer(sae, trainer_config)

    # Create a simple iterator over the layer activations
    def data_iter():
        while True:
            idx = torch.randperm(len(h_layer))[:batch_size]
            yield h_layer[idx]

    print(f"Training for {n_steps} steps...")
    logs = trainer.train(data_iter())
    print(f"Final loss: {logs[-1]['loss']:.4f} | L0: {logs[-1]['l0']:.1f} | Dead: {logs[-1]['n_dead']}")

    # --- Evaluate ---
    true_dirs = get_true_features_at_layer(model, layer)  # [C, D]
    sae_dirs = sae.feature_directions()                    # [F, D]

    # Reconstruction quality
    with torch.no_grad():
        sample = h_layer[:1000].to(device)
        _, _, _ = sae(sample)
        z_sample = sae.encode(sample)
        x_hat_sample = sae.decode(z_sample)
        ev = explained_variance(sample.cpu(), x_hat_sample.cpu())

    # Feature recovery
    recovery = feature_recovery_rate(
        true_dirs=true_dirs.cpu(),
        sae_dirs=sae_dirs.detach().cpu(),
        threshold=0.9
    )

    results = {
        "layer": layer,
        "seed": seed,
        "n_features_sae": n_features_sae,
        "explained_variance": ev,
        "l0": logs[-1]["l0"],
        "n_dead": logs[-1]["n_dead"],
        **recovery,
        "sae_dirs": sae_dirs.detach().cpu(),
    }

    print(f"\nResults:")
    print(f"  Explained variance:  {ev:.3f}")
    print(f"  Recovery rate:       {recovery['recovery_rate']:.3f} ({recovery['n_recovered']}/{recovery['n_true']} features)")
    print(f"  Mean max cos sim:    {recovery['mean_max_cos']:.3f}")
    print(f"  False positive rate: {recovery['false_positive_rate']:.3f}")

    return results


def main():
    results_dir = Path("results/phase1_baselines")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run 3 seeds for cross-seed convergence
    seeds = [42, 123, 777]
    layer = 0  # Test on first layer first
    all_results = []
    all_dirs = []

    for seed in seeds:
        r = run_synthetic_baseline(
            layer=layer,
            n_features_sae=512,
            seed=seed,
            n_steps=10_000,
            batch_size=1024,
        )
        all_results.append(r)
        all_dirs.append(r["sae_dirs"])

    # Cross-seed convergence
    print(f"\n{'='*60}")
    print(f"Cross-Seed Convergence Analysis")
    print(f"{'='*60}")
    conv = cross_seed_convergence(all_dirs, threshold=0.9)
    print(f"Mean matched cosine sim: {conv['mean_match_cos']:.3f}")
    print(f"Convergence rate:        {conv['convergence_rate']:.3f}")

    # --- Phase 1 Gate Check ---
    print(f"\n{'='*60}")
    print(f"PHASE 1 GATE CHECK — Standard SAE Baseline")
    print(f"{'='*60}")

    mean_recovery = np.mean([r["recovery_rate"] for r in all_results])
    print(f"Mean recovery rate: {mean_recovery:.3f}")
    print(f"Gate criterion: >= 0.60")
    print(f"PASS: {mean_recovery >= 0.60}")

    if mean_recovery < 0.60:
        print("\nWARNING: Recovery rate below gate threshold.")
        print("Check: (1) k value, (2) n_steps, (3) l1_coeff, (4) n_features_sae")

    # Save results
    torch.save({
        "all_results": all_results,
        "convergence": conv,
        "mean_recovery": mean_recovery,
    }, results_dir / "synthetic_baseline_results.pt")
    print(f"\nResults saved to {results_dir}/synthetic_baseline_results.pt")


if __name__ == "__main__":
    main()
