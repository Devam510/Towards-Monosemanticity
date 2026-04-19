import torch
from typing import Callable
import torch.nn.functional as F

def causal_intervention_test(
    model_wrapper,
    tokens: torch.Tensor,
    layer_idx: int,
    feature_direction: torch.Tensor,
    intervention_scale: float = 10.0,
    ablation: bool = False
) -> float:
    """
    Tests causal impact of a feature by clamping or ablating its direction in the residual stream
    and measuring KL divergence of the output logits.
    
    Args:
        model_wrapper: PythiaWrapper instance
        tokens: [batch, seq_len] input tokens
        layer_idx: Hook point layer
        feature_direction: [D] Normalized SAE decoder direction
        intervention_scale: Multiplier for clamping (if ablation=False)
        ablation: If true, removes the feature component entirely.
        
    Returns:
        kl_div: Mean KL divergence across the sequence.
    """
    hook_name = model_wrapper.get_layer_name(layer_idx)
    
    # 1. Run baseline un-intervened
    with torch.no_grad():
        base_logits = model_wrapper.model(tokens)
        base_probs = F.softmax(base_logits, dim=-1)
        
    # 2. Define intervention hook
    def intervention_hook(resid_post, hook):
        # resid_post: [batch, seq_len, d_model]
        # feature_direction: [d_model]
        projection = (resid_post @ feature_direction).unsqueeze(-1) * feature_direction
        
        if ablation:
            # Remove the feature component
            return resid_post - projection
        else:
            # Clamp the feature to a high value
            return resid_post + (intervention_scale * feature_direction)
            
    # 3. Form intervened logits
    with torch.no_grad():
        intervened_logits = model_wrapper.model.run_with_hooks(
            tokens, 
            fwd_hooks=[(hook_name, intervention_hook)]
        )
        intervened_probs = F.softmax(intervened_logits, dim=-1)
        
    # 4. Compute KL Div
    # KL(P || Q) = sum(P * log(P/Q))
    kl_div = F.kl_div(
        intervened_probs.log(), 
        base_probs, 
        reduction="batchmean", 
        log_target=False
    )
    
    return kl_div.item()
