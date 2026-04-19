import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pythia_wrapper import PythiaWrapper
from src.injection.inject import FeatureInjector
from src.data.pythia_dataset import PythiaDataset
from src.data.activation_cache import ActivationCache

def inject_and_cache():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = PythiaWrapper(device=device)
    
    injector = FeatureInjector()
    
    block_layer = 1
    # TransformerLens stores Pythia W_out as [d_mlp, d_model]
    W_out = wrapper.model.blocks[block_layer].mlp.W_out.data
    d_model, d_mlp = W_out.shape # Pythia often has [D, D_mlp] or reversed
    print(f"Injecting 5 synthetic features into MLP {block_layer} W_out (shape: {W_out.shape})")
    
    for i in range(5):
        torch.manual_seed(100 + i)
        u = F.normalize(torch.randn(W_out.shape[1], device=device), dim=0)
        v = F.normalize(torch.randn(W_out.shape[0], device=device), dim=0)
        
        # Perturb
        W_out = injector.inject_into_tensor(W_out, u, v, alpha=0.5)
        injector.record_injection(u, v, alpha=0.5, layer=block_layer)
        
    wrapper.model.blocks[block_layer].mlp.W_out.data = W_out
    
    print("Caching perturbed activations...")
    dataset = PythiaDataset(wrapper, seq_len=256)
    layer_indices = [0, 1, 2, 3]
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layer_indices]
    
    cache_dir = "experiments/cache/pythia_70m_perturbed"
    cache = ActivationCache(cache_dir, hook_names, chunk_size=2048)
    
    batch_size = 16
    total_tokens_target = 8192 # test run
    tokens_per_batch = batch_size * dataset.seq_len
    num_batches = total_tokens_target // tokens_per_batch + 1
    
    with torch.no_grad():
        for i in range(num_batches):
            tokens = dataset.get_batch(batch_size, start_idx=i * batch_size).to(device)
            _, run_cache = wrapper.model.run_with_cache(tokens, names_filter=hook_names)
            
            for name in hook_names:
                acts = run_cache[name].flatten(0, 1).cpu()
                if len(acts) > 0:
                    cache.save_batch(name, acts)
                
    print("Perturbed activations cached successfully.")
    
    # Normally we would train an SAE here and run:
    # injector.verify_recovery(sae.feature_directions(), injection_idx=0, direction='output')

if __name__ == "__main__":
    inject_and_cache()
