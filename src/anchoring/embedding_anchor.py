import torch
import torch.nn as nn

def embed_anchor_mse(sae: nn.Module, W_E: torch.Tensor) -> torch.Tensor:
    """
    Computes a boundary anchoring loss aligning SAE features with token embeddings.
    
    Since F (n_features) != V (vocab_size), we compute soft matching:
    We want SAE features (sae.W_dec.T) which have shape [F, d_model]
    and W_E which has shape [V, d_model] to align.
    
    We compute the cosine similarity between each SAE feature and all token embeddings.
    Then we penalize features that don't have AT LEAST ONE strongly aligned token.
    loss = MSE(1.0 - max_cos(features, W_E))
    """
    # W_dec is [d_model, F]. W_dec is already normalized in TopK SAE, but let's be safe.
    f_dirs = nn.functional.normalize(sae.W_dec, dim=0).T  # [F, d_model]
    
    # Normalize token embeddings
    v_dirs = nn.functional.normalize(W_E, dim=1)  # [V, d_model]
    
    # Cosine similarities
    cos_sim = f_dirs @ v_dirs.T  # [F, V]
    
    # For every SAE feature, find its "closest" token embedding
    # We want this max similarity to approach 1.0 (so distance to 1.0 -> 0)
    max_sims, _ = cos_sim.max(dim=1)  # [F]
    
    # L2 penalty on the deviation from perfect alignment
    return ((1.0 - max_sims) ** 2).mean()
