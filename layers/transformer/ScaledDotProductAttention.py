import numpy as np
import torch

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        scalar = np.sqrt(self.d_k)
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar # Q * x^T / (D^0.5)
        assert isinstance(attention_weight, torch.Tensor)
        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_wegith.dim, mask.dim={}, attention_weight.dim={}".format(mask.dim(), attention_weight.dim())
                )
            
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            )
        attention_weight = torch.nn.functional.softmax(attention_weight, dim=2)
        return torch.matmul(attention_weight, v)