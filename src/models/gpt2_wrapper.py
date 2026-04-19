import torch
from transformer_lens import HookedTransformer

class GPT2Wrapper:
    """Wrapper around TransformerLens GPT-2 Small for CCJFR boundary anchoring."""
    
    def __init__(self, model_name="gpt2-small", device="cpu", fp16=False):
        self.device = device
        print(f"Loading {model_name}...")
        dtype = torch.float16 if fp16 else torch.float32
        
        # Load the model — GPT-2-small is 124M params
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype=dtype
        )
        
        self.d_model = self.model.cfg.d_model
        
    def get_layer_name(self, layer_idx: int) -> str:
        """Get the TransformerLens hook name for the residual stream at layer_idx.
        We use hook_resid_pre to align with published Anthropic/jbloom SAEs.
        """
        return f"blocks.{layer_idx}.hook_resid_pre"
        
    def get_W_E(self):
        """Word embedding matrix (dictionary of boundary features)."""
        return self.model.W_E  # [vocab_size, d_model]
        
    def get_W_U(self):
        """Unembedding matrix."""
        return self.model.W_U  # [d_model, vocab_size]
