import torch

def project_to_nullspace(J: torch.Tensor, vec: torch.Tensor, tol: float = 1e-5) -> torch.Tensor:
    """
    Projects vector `vec` into the nullspace of matrix `J`.
    
    Args:
        J: [D_out, D_in]
        vec: [D_in]
        tol: singular value threshold for nullspace
        
    Returns:
        projection: [D_in]
    """
    J_pinv = torch.linalg.pinv(J, rcond=tol)
    P = torch.eye(J.shape[1], device=J.device) - J_pinv @ J
    return P @ vec
    
def batched_nullspace_projection(J_batch: torch.Tensor, vec: torch.Tensor, tol: float = 1e-5) -> torch.Tensor:
    """
    Calculates nullspace projection for a batch of Jacobians.
    J_batch: [batch, D, D]
    vec: [D]
    Returns: [batch, D]
    """
    J_pinv = torch.linalg.pinv(J_batch, rcond=tol)
    I = torch.eye(J_batch.shape[-1], device=J_batch.device).expand_as(J_batch)
    P = I - J_pinv @ J_batch
    
    return (P @ vec.unsqueeze(-1)).squeeze(-1)
