import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.activation_cache import ActivationCache
from src.models.ccjfr import CCJFRConfig, CCJFR
from src.training.ccjfr_trainer import CCJFRTrainer, CCJFRTrainerConfig
from src.anchoring.embedding_anchor import embed_anchor_mse
from src.models.gpt2_wrapper import GPT2Wrapper

def train_gpt2_ccjfr():
    cache_dir = "experiments/cache/gpt2_small"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(cache_dir):
        print(f"Cache dir {cache_dir} not found. Run cache script first.")
        return
        
    wrapper = GPT2Wrapper(device=device, fp16=True)
    
    layer_indices = [0, 1, 2, 3]
    hook_names = [f"blocks.{l}.hook_resid_pre" for l in layer_indices]
    
    cache = ActivationCache(cache_dir, hook_names, chunk_size=2048)
    if cache.n_chunks(hook_names[0]) == 0:
        print("No cache chunks found.")
        return
        
    print(f"Training CCJFR with boundaries on layers 0-3...")

    # We use boundary anchoring over W_E (Embedding)
    W_E = wrapper.get_W_E() # [vocab_size, d_model]
    
    seeds = [42, 43]
    models = []
    
    ckpt_dir = Path("results/gpt2_ccjfr")
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        
        cfg = CCJFRConfig(
            n_layers=4,
            d_model=768,
            n_features=3072, # Expansion factor 4
            l1_coeff=1e-3, 
            consist_coeff=0.2,       # Inter-layer consistency strength
            k=0,
            seed=seed,
            device=device
        )
        ccjfr = CCJFR(cfg, embed_matrix=W_E.to(torch.float32).to(device)).to(device)
        
        train_config = CCJFRTrainerConfig(
            n_steps=2000, 

            batch_size=512,
            lr=3e-4,
            log_every=500,
            device=device
        )
        
        from typing import Dict, Iterator
        def make_loader():
            def gen():
                for _ in range(train_config.n_steps):
                    batch = cache.get_random_batch_all(hook_names, train_config.batch_size, device=device)
                    # Trainer expects {"activations": [h_0, h_1, ...]}
                    ordered_activations = [batch[name] for name in hook_names]
                    yield {"activations": ordered_activations}
            return gen()
            
        trainer = CCJFRTrainer(ccjfr, train_config)
        
        # Computation bounds
        def get_wrapped_block(l):
            def T_l(x):
                # x is [B, D]. Transformer block needs [B, seq_len, D]
                x_3d = x.unsqueeze(1)
                out = wrapper.model.blocks[l](x_3d)
                return out.squeeze(1)
            return T_l
            
        compute_fns = [get_wrapped_block(l) for l in range(3)]
        
        metrics = trainer.train(make_loader(), compute_fns=compute_fns)
        
        last_metric = metrics[-1]
        print(f"Final Step {last_metric['step']} | Total Loss: {last_metric['loss']:.4f}")
        for l in range(4):
            print(f"  Layer {l} | L0: {last_metric['l0_by_layer'][l]:.1f}")
        
        models.append(ccjfr.cpu())
        torch.cuda.empty_cache()
        
        torch.save({
            'cfg': cfg,
            'state_dict': ccjfr.state_dict()
        }, ckpt_dir / f"layers_0-3_seed_{seed}.pt")
        
    # Calculate convergence at Layer 0 (the anchor layer)
    from src.evaluation.ground_truth import cross_seed_convergence
    sae1_layer0_dirs = models[0].saes[0].feature_directions()
    sae2_layer0_dirs = models[1].saes[0].feature_directions()
    
    conv_dict = cross_seed_convergence([sae1_layer0_dirs, sae2_layer0_dirs])
    conv_0 = conv_dict["convergence_rate"]
    print(f"\nCCJFR Layer 0 Convergence (GPT-2 Small): {conv_0:.4f}")
    
    # Save result
    with open("results/gpt2_ccjfr_convergence.txt", "w") as f:
        f.write(str(conv_0))

if __name__ == "__main__":
    train_gpt2_ccjfr()
