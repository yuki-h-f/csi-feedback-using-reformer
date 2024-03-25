from typing import Any
import torch

class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, quantization_bits: int) -> torch.Tensor:
        x = torch.floor(x * 2 ** quantization_bits) / 2 ** quantization_bits
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        tensor, quantization_bits = inputs
        ctx.quntization_bits = quantization_bits

    @staticmethod
    def backward(ctx, grad_output) -> Any:
        return grad_output, None

class _Quantizer(torch.nn.Module):
    def __init__(self, quantization_bits: int = 4) -> None:
        super().__init__()
        self.quantization_bits = quantization_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.floor(x * 2 ** self.quantization_bits) / 2 ** self.quantization_bits
        return x