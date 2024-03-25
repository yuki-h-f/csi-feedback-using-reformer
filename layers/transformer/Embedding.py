import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, d_model: int, pad_idx: int = 0 ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(32, d_model)
    
    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.linear(input_batch)