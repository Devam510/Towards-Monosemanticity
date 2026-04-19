import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.activation_cache import ActivationCache
from src.models.sae import SAEConfig, SparseAutoencoder
from src.training.trainer import SAETrainer, TrainerConfig

def train_gpt2_baseline_saes():
    cache_dir = "experiments/cache/gpt2_small"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(cache_dir):
        print(f"Cache dir {cache_dir} not found. Run scripts/cache_activations.py first.")
        return
        
    hook_name = "blocks.0.hook_resid_pre"  # We primarily test identifiability on a chosen layer
    
    cache = ActivationCache(cache_dir, [hook_name], chunk_size=2048)
    if cache.n_chunks(hook_name) == 0:
        print(f"No chunks cached for {hook_name}")
        return
            
    print(f"Found cache chunks for {hook_name}. Training GPT-2 Baseline SAEs...")

    seeds = [42, 43]
    models = []
    
    # Checkpoint dir
    ckpt_dir = Path("results/gpt2_saes")
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        sae_config = SAEConfig(
            d_model=768,      # GPT-2 Small d_model
            n_features=3072,  # 4x expansion factor
            l1_coeff=1e-3, 
            k=0,              # L1 mode
            lr=3e-4,
            seed=seed,
            device=device
        )
        sae = SparseAutoencoder(sae_config).to(device)
        
        train_config = TrainerConfig(
            n_steps=5000, 
            batch_size=512,  # Safe batch size for 4GB
            log_every=500,
            device=device
        )
        
        def make_loader():
            return cache.iter_random_batches(hook_name, batch_size=train_config.batch_size, n_steps=train_config.n_steps, device=device)
            
        trainer = SAETrainer(sae, train_config)
        metrics = trainer.train(make_loader())
        
        last_metric = metrics[-1]
        print(f"Final Step {last_metric['step']} | Loss: {last_metric['loss']:.4f} | Recon: {last_metric['recon_mse']:.4f} | L0: {last_metric['l0']:.1f}")
        
        models.append(sae)
        
        # Save model weights
        torch.save({
            'cfg': sae_config,
            'state_dict': sae.state_dict()
        }, ckpt_dir / f"layer_0_seed_{seed}.pt")
        
    # Calculate cross-seed convergence
    from src.evaluation.ground_truth import cross_seed_convergence
    sae1_dirs = models[0].feature_directions()
    sae2_dirs = models[1].feature_directions()
    
    convergence_dict = cross_seed_convergence([sae1_dirs, sae2_dirs])
    convergence = convergence_dict["convergence_rate"]
    print(f"\n✅ Standard SAE Cross-Seed Convergence (GPT-2 Small, Layer 0): {convergence:.4f}")
    
    # Save result
    with open("results/gpt2_baseline_convergence.txt", "w") as f:
        f.write(str(convergence))

if __name__ == "__main__":
    train_gpt2_baseline_saes()
