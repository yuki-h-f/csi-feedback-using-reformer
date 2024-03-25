import torch
from .UeSideCompressor import UeSideCompressor
from .BsSideReconstructor import BsSideReconstructor

class CsiFeedbackTransformerAutoEncoder(torch.nn.Module):
    def __init__(self, num_embeddings: int, max_len: int, pad_idx: int, d_model: int, N: int, d_ff: int, heads_num: int, dropout_rate: float, layer_norm_eps: float, device: torch.device = torch.device("cpu"), code_rate: int = 1, use_quantizer: bool = True, quantization_bits: int = 4) -> None:
        super().__init__()
        self.compressor = UeSideCompressor(num_embeddings, max_len, pad_idx, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps, device, code_rate, use_quantizer, quantization_bits)
        self.reconstructor = BsSideReconstructor(num_embeddings, max_len, pad_idx, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps, device, code_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compressor(x)
        x = self.reconstructor(x) 
        return x