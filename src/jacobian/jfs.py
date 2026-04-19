import torch
from .nullspace import batched_nullspace_projection

def compute_jfs(J_batch: torch.Tensor, feature_directions: torch.Tensor, tol: float = 1e-5) -> torch.Tensor:
    """
    Computes JFS for multiple features over a batch of Jacobians.
    
    Args:
        J_batch: [batch, D, D]
        feature_directions: [F, D]
        
    Returns:
        jfs_scores: [F]
    """
    F, D = feature_directions.shape
    scores = torch.zeros(F, device=J_batch.device)
    orig_norm = torch.norm(feature_directions, dim=-1) # [F]
    
    for i in range(F):
        d_f = feature_directions[i]
        
        if orig_norm[i] < 1e-8:
            scores[i] = 0.0
            continue
            
        proj = batched_nullspace_projection(J_batch, d_f, tol)
        proj_norm = torch.norm(proj, dim=-1)  # [batch]
        
        null_ratio = (proj_norm / orig_norm[i]).mean()
        scores[i] = 1.0 - null_ratio
        
    return scores
