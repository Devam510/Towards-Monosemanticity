import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gpt2_wrapper import GPT2Wrapper
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
    
    # Force CUDA if available, but memory is limited (4GB VRAM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load GPT-2 Small in fp16 to save memory
    wrapper = GPT2Wrapper(device=device, fp16=True)
    
    # Sequence length 64 is safe for 4GB VRAM
    dataset = PythiaDataset(wrapper, seq_len=64)
    
    # We will cache layers 0, 1, 2, 3, 4 for the joint training
    layer_indices = [0, 1, 2, 3, 4]
    hook_names = [wrapper.get_layer_name(l) for l in layer_indices]
    
    # Init cache
    cache_dir = "experiments/cache/gpt2_small"
    cache = ActivationCache(cache_dir, hook_names, chunk_size=2048)
    
    # 32 batch size with 64 seq_len = 2048 tokens per batch
    # We want 500 batches to get ~1M tokens
    batch_size = 32
    num_batches = 500
    
    print(f"Caching GPT-2 activations: {num_batches} batches of size {batch_size} (seq_len=64)")
    
    # Accumulators for chunks
    buffer = {name: [] for name in hook_names}
    
    # Free up memory before running
    if device == "cuda":
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            tokens = dataset.get_batch(batch_size)
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

    print(f"Done caching activations to {cache_dir}")

if __name__ == "__main__":
    main()
