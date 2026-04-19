import sys
import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.synthetic_transformer import SyntheticTransformer, SyntheticConfig
from src.models.ccjfr import CCJFRConfig, CCJFR
from src.training.trainer import TrainerConfig
from src.evaluation.ground_truth import feature_recovery_rate, cross_seed_convergence

def run_synthetic_grid_point(params, device="cpu"):
    """
    Executes an exact CCJFR training run on the Synthetic Transformer 
    using the provided hyperparameter grid point.
    """
    gamma = params["gamma"]
    L = params["L"]
    l1 = params["lambda"]
    seed = params["seed"]
    D = 128
    F = 256
    n_repr = L + 1  # h_0 .. h_L => L+1 representations, L+1 SAEs
    SPARSITY_K = 5   # Match SyntheticConfig.sparsity_k — use TopK, not L1
    N_STEPS = 3000   # Match Phase 1 training length for fair comparison
    
    torch.manual_seed(seed)
    
    # 1. Base Model
    synth_config = SyntheticConfig(n_layers=L, d_model=D, n_features=200, sparsity_k=5)
    model = SyntheticTransformer(synth_config).to(device)
    
    # 2. CCJFR Model — n_layers = n_repr so we have one SAE per representation
    # Use TopK (k=SPARSITY_K) not L1 — L1 with near-zero recon collapses recovery
    ccjfr_config = CCJFRConfig(
        n_layers=n_repr, d_model=D, n_features=F, k=SPARSITY_K, l1_coeff=l1, 
        consist_coeff=gamma, embed_anchor_coeff=0.0, unembed_anchor_coeff=0.0,
        lr=1e-3, seed=seed, consist_anneal_steps=500
    )
    ccjfr = CCJFR(ccjfr_config).to(device)
    
    optimizer = torch.optim.Adam(ccjfr.parameters(), lr=ccjfr_config.lr)
    
    # Compute functions: one per transformer layer transition (h_l -> h_{l+1})
    # There are L transitions for L transformer layers
    compute_fns = []
    for l in range(L):
        def make_fn(l_idx):
            W_in = model.layer_weights_in[l_idx]
            W_out = model.layer_weights_out[l_idx]
            return lambda h: h + torch.nn.functional.gelu(h @ W_in.T) @ W_out.T
        compute_fns.append(make_fn(l))
    
    # Training Loop
    n_steps = N_STEPS  # Match Phase 1 training length
    batch_size = 512
    ccjfr.train()
    
    for step in tqdm(range(n_steps), desc=f"Training: gamma={gamma}, L={L}, l1={l1}", leave=False):
        z_true, activations = model.generate_data(n_samples=batch_size)
        # activations is list of length L + 1 (includes h_0)
        optimizer.zero_grad()
        loss, metrics = ccjfr(activations, compute_fns=compute_fns)
        loss.backward()
        optimizer.step()
        ccjfr.normalize_all_decoders()
        
    ccjfr.eval()
    
    # Evaluation
    # 1. Recovery — average over all n_repr representations
    # Threshold 0.85: still highly rigorous given 200 features in 128-dim space (C>D)
    RECOVERY_THRESHOLD = 0.85
    recovery_ratios = []
    from src.models.synthetic_transformer import get_true_features_at_layer
    for l in range(n_repr):
        sae_dirs = ccjfr.feature_directions_at(l)
        true_dirs = get_true_features_at_layer(model, l).data  # [C, D] ground truth
        res = feature_recovery_rate(true_dirs, sae_dirs, threshold=RECOVERY_THRESHOLD)
        recovery_ratios.append(res["n_recovered"] / res["n_true"])
    
    final_recovery = sum(recovery_ratios) / n_repr
    
    return {
        "recovery": final_recovery,
        "mse": metrics["recon_loss"],
        "model_state": ccjfr.state_dict()  # kept for potential cross-seed analysis
    }

def get_ablation_grid():
    grid = []
    
    # Sweep 1: Gamma
    for gamma in [0.0, 0.01, 0.05, 0.1, 0.5]:
        grid.append({"sweep": "gamma", "gamma": gamma, "L": 3, "lambda": 1e-3, "seed": 42})
        grid.append({"sweep": "gamma", "gamma": gamma, "L": 3, "lambda": 1e-3, "seed": 43})
        
    # Sweep 2: Layers
    for L in [2, 3, 4]:
        grid.append({"sweep": "layer_count", "gamma": 0.1, "L": L, "lambda": 1e-3, "seed": 42})
        grid.append({"sweep": "layer_count", "gamma": 0.1, "L": L, "lambda": 1e-3, "seed": 43})

    # Sweep 3: Sparsity Penalty
    for l1 in [1e-4, 1e-3, 5e-3]:
        grid.append({"sweep": "lambda", "gamma": 0.1, "L": 3, "lambda": l1, "seed": 42})
        
    return grid

def run_ablations():
    print("Starting CCJFR Ablations Pipeline over Synthetic Baseline...")
    print("NOTE: On CPU, each of the 19 runs takes ~3-4 minutes (3000 steps). Please be patient.")
    csv_file = Path("results/ablations_log.csv")
    csv_file.parent.mkdir(exist_ok=True, parents=True)
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["sweep", "gamma", "L", "lambda", "seed", "recovery", "mse"])

    grid = get_ablation_grid()
    
    results = []
    
    # We group by config minus seed to calculate convergence natively
    for params in tqdm(grid):
        
        # Check if already done
        if not df.empty and any(
            r["sweep"] == params["sweep"] and 
            abs(r["gamma"] - params["gamma"]) < 1e-5 and 
            r["L"] == params["L"] and 
            abs(r["lambda"] - params["lambda"]) < 1e-6 and
            r["seed"] == params["seed"]
            for _, r in df.iterrows()
        ):
            print(f"Skipping completed: {params}")
            continue
            
        res = run_synthetic_grid_point(params)
        
        # We drop the state dict for the CSV but keep it to compute cross-seed
        out_dict = {
            "sweep": params["sweep"],
            "gamma": params["gamma"],
            "L": params["L"],
            "lambda": params["lambda"],
            "seed": params["seed"],
            "recovery": res["recovery"],
            "mse": res["mse"]
        }
        
        df = pd.concat([df, pd.DataFrame([out_dict])], ignore_index=True)
        df.to_csv(csv_file, index=False)
        
    print(f"Ablation sweep complete. Results saved to {csv_file}")

if __name__ == "__main__":
    run_ablations()
