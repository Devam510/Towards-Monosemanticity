import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pythia_wrapper import PythiaWrapper
from src.evaluation.causal import causal_intervention_test
from src.models.sae import SAEConfig, SparseAutoencoder

def run_causal_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = PythiaWrapper(device=device)
    
    # Dummy prompt to run through model for causal testing
    text = "The quick brown fox jumps over the lazy dog."
    tokens = wrapper.model.to_tokens(text).to(device)
    
    layer_idx = 0
    config = SAEConfig(d_model=512, n_features=256, device=device)
    
    # Normally we load trained CCJFR and trained Baseline here
    ccjfr_sae = SparseAutoencoder(config).to(device)
    base_sae = SparseAutoencoder(config).to(device)
    
    print("Evaluating 10 Random Features on CCJFR vs Baseline (Clamping & Ablation KL Divergence)")
    
    ccjfr_kls = []
    base_kls = []
    
    for i in range(10):
        # Clamping Test
        ccjfr_dir = ccjfr_sae.feature_directions()[i]
        base_dir = base_sae.feature_directions()[i]
        
        kl_ccjfr = causal_intervention_test(wrapper, tokens, layer_idx, ccjfr_dir, intervention_scale=5.0, ablation=False)
        kl_base = causal_intervention_test(wrapper, tokens, layer_idx, base_dir, intervention_scale=5.0, ablation=False)
        
        ccjfr_kls.append(kl_ccjfr)
        base_kls.append(kl_base)
        
        print(f"Feature {i:2d} | CCJFR KL: {kl_ccjfr:.4f} | Base KL: {kl_base:.4f}")
        
    print(f"\nMean CCJFR KL Divergence: {sum(ccjfr_kls)/len(ccjfr_kls):.4f}")
    print(f"Mean Baseline KL Divergence: {sum(base_kls)/len(base_kls):.4f}")
    print("Higher KL indicates the feature intervention more significantly steered the model's logits.")

if __name__ == "__main__":
    run_causal_evaluation()
