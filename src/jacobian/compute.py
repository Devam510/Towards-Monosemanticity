import torch
from torch.func import jacrev, vmap
from typing import Callable

def compute_batched_jacobian(compute_fn: Callable, x: torch.Tensor) -> torch.Tensor:
    """
    Computes batched Jacobian of compute_fn(x) with respect to x.
    
    Args:
        compute_fn: Function mapping [1, D] -> [1, D]
        x: Tensor of shape [batch, D]
        
    Returns:
        Jacobians: Tensor of shape [batch, D_out, D_in]
    """
    def single_fn(x_i):
        # x_i is [D]
        return compute_fn(x_i.unsqueeze(0)).squeeze(0)
        
    jacobian_fn = vmap(jacrev(single_fn))
    return jacobian_fn(x)
