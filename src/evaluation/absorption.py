import torch

def find_absorbing_features(
    z_true: torch.Tensor, 
    z_sae: torch.Tensor, 
    threshold: float = 0.5
) -> dict:
    """
    Evaluates Feature Absorption.
    If multiple independent ground truth features trigger the same SAE feature,
    the SAE is suffering from "absorption".
    
    Args:
        z_true: [batch, F_true] ground truth activations
        z_sae: [batch, F_sae] learned activations
        threshold: Correlation threshold
        
    Returns:
        Dict detailing absorption statistics.
    """
    # Calculate correlations between true features and learned features
    z_true_norm = z_true - z_true.mean(0)
    z_sae_norm = z_sae - z_sae.mean(0)
    
    z_true_std = z_true_norm.std(0) + 1e-8
    z_sae_std = z_sae_norm.std(0) + 1e-8
    
    corr = (z_true_norm.T @ z_sae_norm) / (z_true.shape[0] * z_true_std.unsqueeze(1) * z_sae_std.unsqueeze(0))
    
    # Analyze how many true features map strongly to the same SAE feature
    absorption_count = 0
    sae_mappings = (corr > threshold).sum(dim=0)
    absorbed_features = (sae_mappings > 1).sum().item()
    
    return {
        "absorbed_sae_features": absorbed_features,
        "total_sae_features": z_sae.shape[1]
    }
