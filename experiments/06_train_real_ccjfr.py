import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.activation_cache import ActivationCache
from src.models.ccjfr import CCJFRConfig, CCJFR
from src.models.pythia_wrapper import PythiaWrapper
from src.training.trainer import TrainerConfig

def train_real_ccjfr():
    cache_dir = "experiments/cache/pythia_70m_res"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(cache_dir):
        print(f"Cache dir {cache_dir} not found. Run scripts/cache_activations.py first")
        return
        
    layer_indices = [0, 1, 2, 3]
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layer_indices]
    
    cache = ActivationCache(cache_dir, hook_names, chunk_size=2048)
    for name in hook_names:
        if cache.n_chunks(name) == 0:
            print(f"No chunks cached for {name}")
            return
            
    # Load Pythia Wrapper to get W_E, W_U and model blocks for T_l
    wrapper = PythiaWrapper(device=device)
    wrapper.model.eval()
    
    W_E = wrapper.get_embedding_weight()
    W_U = wrapper.get_unembedding_weight()
    
    d_model = 512
    n_features = 2048 # 4x overcomplete for 4GB VRAM constraint
    batch_size = 1024
    n_steps = 100
    
    config = CCJFRConfig(
        n_layers=4,
        d_model=d_model,
        n_features=n_features,
        k=0, # L1 mode
        l1_coeff=1e-3,
        consist_coeff=0.1,
        embed_anchor_coeff=0.1,
        unembed_anchor_coeff=0.0, # skip unembed for now
        lr=3e-4,
        seed=42,
        device=device,
        consist_anneal_steps=500
    )
    
    ccjfr = CCJFR(config, embed_matrix=W_E, unembed_matrix=W_U).to(device)
    
    optimizer = torch.optim.Adam(ccjfr.parameters(), lr=config.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=300)
    
    compute_fns = []
    for l in layer_indices[:-1]:
        block = wrapper.model.blocks[l+1]
        def make_fn(blk):
            def fn(h):
                return blk(h.unsqueeze(1)).squeeze(1)
            return fn
        compute_fns.append(make_fn(block))
        
    print(f"\n--- Training CCJFR on layers {layer_indices} ---")
    
    ccjfr.train()
    
    for step in range(n_steps):
        # We need to sample synchronously across all layers
        chunk_idx = torch.randint(0, cache.n_chunks(hook_names[0]), (1,)).item()
        
        # sample a random sub-batch from this chunk
        sub_idx = torch.randperm(cache.chunk_size)[:batch_size]
        
        activations = []
        for name in hook_names:
            chunk = cache.load_chunk(name, chunk_idx).to(device)
            if len(chunk) < batch_size:
                activations.append(chunk)
            else:
                activations.append(chunk[sub_idx])
                
        optimizer.zero_grad()
        loss, metrics = ccjfr(activations, compute_fns=compute_fns, actual_logits=None)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ccjfr.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ccjfr.normalize_all_decoders()
        
        if step % 25 == 0 or step == n_steps - 1:
            l = metrics['total_loss']
            r = metrics['recon_loss']
            c = metrics['consist_loss']
            e = metrics['embed_anchor_loss']
            print(f"Step {step:4d} | Total: {l:.4f} | Recon: {r:.4f} | Consist: {c:.4f} | Embed: {e:.4f}")

    print("Completed CCJFR training.")

if __name__ == "__main__":
    train_real_ccjfr()
