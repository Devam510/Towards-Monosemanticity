import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pythia_wrapper import PythiaWrapper
from src.jacobian.compute import compute_batched_jacobian
from src.jacobian.jfs import compute_jfs
from src.models.sae import SAEConfig, SparseAutoencoder
from src.data.activation_cache import ActivationCache

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wrapper = PythiaWrapper(device=device)
    
    cache_dir = "experiments/cache/pythia_70m_res"
    hook_name = "blocks.0.hook_resid_post"
    cache = ActivationCache(cache_dir, [hook_name], chunk_size=2048)
    if cache.n_chunks(hook_name) == 0:
        print("No cache chunks found.")
        return
        
    x = cache.load_chunk(hook_name, 0)[:32].to(device) # evaluate on small batch to save memory
    
    block = wrapper.model.blocks[1]
    def compute_fn(h):
        return block(h.unsqueeze(1)).squeeze(1)
        
    print(f"Computing batched Jacobian for {len(x)} inputs...")
    J_batch = compute_batched_jacobian(compute_fn, x)
    print(f"J_batch shape: {J_batch.shape}")
    
    print("Testing JFS on SAE features")
    config = SAEConfig(d_model=512, n_features=256, device=device) # test with 256 features
    sae = SparseAutoencoder(config).to(device)
    
    jfs = compute_jfs(J_batch, sae.feature_directions(), tol=1e-5)
    print(f"JFS computed successfully for {len(jfs)} features. Mean JFS: {jfs.mean().item():.4f}")

if __name__ == "__main__":
    run()
