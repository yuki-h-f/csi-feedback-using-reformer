import torch

from .transformer.Encoder import TransformerEncoder

class BsSideReconstructor(torch.nn.Module):
    def __init__(self, num_embeddings: int, max_len: int, pad_idx: int, d_model: int, N: int, d_ff: int, heads_num: int, dropout_rate: float, layer_norm_eps: float, device: torch.device = torch.device("cpu"), code_rate: int = 1) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(int(32 * 64 * code_rate), 32 * 64)
        self.encoder = TransformerEncoder(num_embeddings, max_len, pad_idx, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps, device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.reshape(x, (x.size(0), 64, 32))
        x = self.encoder(x)
        return x
    
    def backward(self, x):
        print("call backward")
        return x
