import torch
from .transformer.Encoder import TransformerEncoder
from .transformer.Quantizer import Quantizer

class UeSideCompressor(torch.nn.Module):
    def __init__(self, num_embeddings: int, max_len: int, pad_idx: int, d_model: int, N: int, d_ff: int, heads_num: int, dropout_rate: float, layer_norm_eps: float, device: torch.device = torch.device("cpu"), code_rate: int = 1, use_quantizer: bool = True, quantization_bits: int = 4) -> None:
        super().__init__() 
        self.encoder = TransformerEncoder(num_embeddings, max_len, pad_idx, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps, device)
        self.linear = torch.nn.Linear(32 * 64, int(32 * 64 * code_rate))
        self.quantizer = Quantizer.apply
        self.use_quantizer = use_quantizer
        self.quantization_bits = quantization_bits
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = self.quantizer(x, self.quantization_bits)  if self.use_quantizer else x
        return x

