"""
Experiment 02: CCJFR on Synthetic Data
Phase 1 — Step 1.4

Trains CCJFR with and without boundary anchoring to see if it fixes
the cross-seed convergence issue.

Expected result: 
- CCJFR (no anchoring) >= 90% recovery
- CCJFR (anchoring) >= 95% recovery
- Cross-seed convergence >= 0.97
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.models.synthetic_transformer import SyntheticTransformer, SyntheticConfig, get_true_features_at_layer
from src.models.ccjfr import CCJFR, CCJFRConfig
from src.training.ccjfr_trainer import CCJFRTrainer, CCJFRTrainerConfig
from src.evaluation.ground_truth import feature_recovery_rate, cross_seed_convergence

def run_synthetic_ccjfr(
    seed: int = 42,
    use_anchoring: bool = False,
    n_features_sae: int = 512,
    n_steps: int = 3_000,
    batch_size: int = 1024,
    device: str = None
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"CCJFR | Anchoring: {use_anchoring} | Seed {seed}")
    print(f"{'='*60}")

    # Build synthetic model
    syn_config = SyntheticConfig(
        n_features=100,
        d_model=256,
        n_layers=3, 
        sparsity_k=5,
        n_samples=50_000,
        seed=0
    )
    model = SyntheticTransformer(syn_config).to(device)
    
    # Generate data
    z_true, activations = model.generate_data(n_samples=50_000, seed=seed)
    
    # Compute actual logits
    with torch.no_grad():
        actual_logits_all = activations[-1].to(device) @ model.unembed_matrix.T.to(device)

    # CCJFR Config - We have (n_layers + 1) representations (h_0 to h_3)
    n_representations = syn_config.n_layers + 1
    
    ccjfr_config = CCJFRConfig(
        n_layers=n_representations,
        d_model=syn_config.d_model,
        n_features=n_features_sae,
        k=syn_config.sparsity_k,
        l1_coeff=1e-4,
        consist_coeff=0.1,
        embed_anchor_coeff=0.1 if use_anchoring else 0.0,
        unembed_anchor_coeff=0.1 if use_anchoring else 0.0,
        lr=2e-4,
        seed=seed,
        device=device,
        consist_anneal_steps=1000
    )

    ccjfr = CCJFR(
        config=ccjfr_config,
        embed_matrix=model.embed_matrix.to(device) if use_anchoring else None,
        unembed_matrix=model.unembed_matrix.to(device) if use_anchoring else None
    ).to(device)

    trainer_config = CCJFRTrainerConfig(
        n_steps=n_steps,
        batch_size=batch_size,
        lr=2e-4,
        log_every=500,
        device=device
    )
    trainer = CCJFRTrainer(ccjfr, trainer_config)

    # Compute functions
    compute_fns = []
    for l in range(syn_config.n_layers):  # l = 0, 1, 2
        def make_fn(l=l):
            W_in = model.layer_weights_in[l].to(device)
            W_out = model.layer_weights_out[l].to(device)
            return lambda h: h + torch.nn.functional.gelu(h @ W_in.T) @ W_out.T
        compute_fns.append(make_fn())

    def data_iter():
        n_samples = len(activations[0])
        while True:
            idx = torch.randperm(n_samples)[:batch_size]
            yield {
                "activations": [act[idx] for act in activations],
                "logits": actual_logits_all[idx] if use_anchoring else None
            }

    print("Training...")
    logs = trainer.train(data_iter(), compute_fns=compute_fns)
    
    results = {}
    for l in range(n_representations):
        true_dirs = get_true_features_at_layer(model, l)
        sae_dirs = ccjfr.feature_directions_at(l)
        
        recovery = feature_recovery_rate(
            true_dirs=true_dirs.to(device),
            sae_dirs=sae_dirs.to(device),
            threshold=0.9
        )
        results[f"layer_{l}"] = {
            "recovery": recovery["recovery_rate"],
            "dirs": sae_dirs.detach().cpu()
        }
        print(f"Layer {l} Recovery: {recovery['recovery_rate']:.3f} | False Pos Rate: {recovery['false_positive_rate']:.3f}")

    return results

def main():
    results_dir = Path("results/phase1_ccjfr")
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123]
    all_results_no_anchor = []
    all_results_anchor = []

    # No anchoring
    for seed in seeds:
        res = run_synthetic_ccjfr(seed=seed, use_anchoring=False)
        all_results_no_anchor.append(res)

    # With anchoring
    for seed in seeds:
        res = run_synthetic_ccjfr(seed=seed, use_anchoring=True)
        all_results_anchor.append(res)

    # Evaluate NO ANCHOR
    print("\n" + "="*60)
    print("RESULTS: CCJFR (No Anchoring)")
    l0_recov_no = np.mean([r["layer_0"]["recovery"] for r in all_results_no_anchor])
    print(f"Mean Recovery (Layer 0): {l0_recov_no:.3f} (Gate >= 0.90: {l0_recov_no >= 0.90})")
    
    dirs_no = [r["layer_0"]["dirs"] for r in all_results_no_anchor]
    conv_no = cross_seed_convergence(dirs_no, threshold=0.9)
    print(f"Cross-seed convergence: {conv_no['convergence_rate']:.3f}")

    # Evaluate ANCHOR
    print("\n" + "="*60)
    print("RESULTS: CCJFR (With Anchoring)")
    l0_recov_y = np.mean([r["layer_0"]["recovery"] for r in all_results_anchor])
    print(f"Mean Recovery (Layer 0): {l0_recov_y:.3f} (Gate >= 0.95: {l0_recov_y >= 0.95})")

    dirs_y = [r["layer_0"]["dirs"] for r in all_results_anchor]
    conv_y = cross_seed_convergence(dirs_y, threshold=0.9)
    print(f"Cross-seed convergence: {conv_y['convergence_rate']:.3f} (Gate >= 0.97: {conv_y['convergence_rate'] >= 0.97})")

    torch.save({
        "no_anchor": all_results_no_anchor,
        "anchor": all_results_anchor
    }, results_dir / "ccjfr_results.pt")

if __name__ == "__main__":
    main()
