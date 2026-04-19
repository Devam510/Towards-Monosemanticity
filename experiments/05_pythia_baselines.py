import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.activation_cache import ActivationCache
from src.models.sae import SAEConfig, SparseAutoencoder
from src.training.trainer import SAETrainer, TrainerConfig

def train_baseline_saes():
    cache_dir = "experiments/cache/pythia_70m_res"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(cache_dir):
        print(f"Cache dir {cache_dir} not found. Run scripts/cache_activations.py first.")
        return
        
    layer_indices = [0, 1, 2, 3]
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layer_indices]
    
    # Rely on ActivationCache to auto-detect existing chunks
    cache = ActivationCache(cache_dir, hook_names, chunk_size=2048)
    for name in hook_names:
        if cache.n_chunks(name) == 0:
            print(f"No chunks cached for {name}")
            return
            
    print("Found cache chunks for layers:", hook_names)

    # For Phase 2 baseline test across seeds for Layer 0
    test_layer_name = hook_names[0]
    seeds = [42]
    
    print(f"\n--- Training baseline SAE for {test_layer_name} ---")
    for seed in seeds:
        print(f"Seed {seed} | L1 coeff = 1e-3")
        sae_config = SAEConfig(
            d_model=512,
            n_features=2048,
            l1_coeff=1e-3, 
            k=0,           # L1 mode
            lr=3e-4,
            seed=seed,
            device=device
        )
        sae = SparseAutoencoder(sae_config).to(device)
        
        train_config = TrainerConfig(
            n_steps=200, 
            batch_size=1024,
            log_every=50,
            device=device
        )
        
        # Generator for trainer
        def make_loader():
            return cache.iter_random_batches(test_layer_name, batch_size=train_config.batch_size, n_steps=train_config.n_steps, device=device)
            
        trainer = SAETrainer(sae, train_config)
        metrics = trainer.train(make_loader())
        
        last_metric = metrics[-1]
        print(f"Final Step {last_metric['step']} | Loss: {last_metric['loss']:.4f} | Recon: {last_metric['recon_mse']:.4f} | EV: {last_metric['explained_var']:.4f} | L0: {last_metric['l0']:.1f}")

if __name__ == "__main__":
    train_baseline_saes()
