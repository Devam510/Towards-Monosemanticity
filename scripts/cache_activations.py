import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pythia_wrapper import PythiaWrapper
from src.data.pythia_dataset import PythiaDataset
from src.data.activation_cache import ActivationCache

def ensure_installed():
    try:
        import datasets
    except ImportError:
        import subprocess
        print("Installing datasets...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])

def main():
    ensure_installed()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = PythiaWrapper(device=device)
    dataset = PythiaDataset(wrapper, seq_len=256)
    
    layer_indices = [0, 1, 2, 3]
    hook_names = [wrapper.get_layer_name(l) for l in layer_indices]
    
    # Init cache
    cache_dir = "experiments/cache/pythia_70m_res"
    cache = ActivationCache(cache_dir, hook_names, chunk_size=2048)
    
    # Reduce tokens for quick verification of stability (Phase 2 testing)
    batch_size = 16
    total_tokens_target = 8192
    tokens_per_batch = batch_size * dataset.seq_len
    num_batches = total_tokens_target // tokens_per_batch + 1
    
    print(f"Caching activations for {total_tokens_target} tokens ({num_batches} batches of size {batch_size})")
    
    # Accumulators for chunks
    buffer = {name: [] for name in hook_names}
    
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            tokens = dataset.get_batch(batch_size, start_idx=i * batch_size)
            tokens = tokens.to(device)
            
            # run_with_cache
            _, run_cache = wrapper.model.run_with_cache(tokens, names_filter=hook_names)
            
            for name in hook_names:
                acts = run_cache[name] # [batch, seq_len, d_model]
                flattened = acts.flatten(0, 1) # [batch * seq_len, d_model]
                buffer[name].append(flattened.cpu())
                
                # Check if buffer has enough for a chunk
                curr_cat = torch.cat(buffer[name], dim=0)
                while len(curr_cat) >= cache.chunk_size:
                    chunk_to_save = curr_cat[:cache.chunk_size]
                    cache.save_batch(name, chunk_to_save)
                    curr_cat = curr_cat[cache.chunk_size:]
                buffer[name] = [curr_cat]
                
    # Save the remaining
    for name in hook_names:
        if len(buffer[name]) > 0:
            curr_cat = torch.cat(buffer[name], dim=0)
            if len(curr_cat) > 0:
                cache.save_batch(name, curr_cat)

    print("Done caching activations.")

if __name__ == "__main__":
    main()
