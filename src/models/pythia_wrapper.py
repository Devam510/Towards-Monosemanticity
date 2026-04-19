import torch
from transformer_lens import HookedTransformer

class PythiaWrapper:
    """Wrapper for Pythia-70M using TransformerLens to extract activations and weights."""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Pythia-70M on {self.device}...")
        self.model = HookedTransformer.from_pretrained("pythia-70m-deduped", device=self.device)
        self.model.eval()
        
    def get_embedding_weight(self) -> torch.Tensor:
        """Returns W_E (d_vocab, d_model)"""
        return self.model.W_E.detach()
        
    def get_unembedding_weight(self) -> torch.Tensor:
        """Returns W_U (d_model, d_vocab)"""
        return self.model.W_U.detach()
        
    def get_layer_name(self, layer_idx: int) -> str:
        """Returns hook name for residual stream of given layer."""
        return f"blocks.{layer_idx}.hook_resid_post"
