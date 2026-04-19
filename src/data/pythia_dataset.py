import torch
from typing import List

class PythiaDataset:
    """Loads and tokenizes a text dataset for Pythia."""
    
    def __init__(self, model_wrapper, dataset_name: str = "NeelNanda/pile-10k", seq_len: int = 256):
        import datasets
        self.model = model_wrapper.model
        print(f"Loading dataset {dataset_name}...")
        self.dataset = datasets.load_dataset(dataset_name, split="train")
        self.seq_len = seq_len
        self.current_idx = 0
        self.current_chunks = []
        
    def get_batch(self, batch_size: int) -> torch.Tensor:
        """Gets a batch of tokens, padded/truncated to seq_len."""
        tokens_list = []
        
        while len(tokens_list) < batch_size:
            # If we don't have enough chunks, fetch the next document
            if len(self.current_chunks) == 0:
                if self.current_idx >= len(self.dataset):
                    raise ValueError(f"End of dataset reached at index {self.current_idx}")
                
                text = self.dataset[self.current_idx]["text"]
                toks = self.model.to_tokens(text)[0]
                
                # Extract all possible chunks from this text
                for i in range(0, len(toks) - self.seq_len + 1, self.seq_len):
                    self.current_chunks.append(toks[i:i + self.seq_len])
                
                self.current_idx += 1
            
            # Append available chunks
            if len(self.current_chunks) > 0:
                tokens_list.append(self.current_chunks.pop(0))
            
        return torch.stack(tokens_list)

