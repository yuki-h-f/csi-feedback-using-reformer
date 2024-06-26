import torch
from layers.transformer.ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h

        self.W_k = torch.nn.Parameter(
            torch.Tensor(h, d_model, self.d_k)
        )
        self.W_q = torch.nn.Parameter(
            torch.Tensor(h, d_model, self.d_k)
        )
        self.W_v = torch.nn.Parameter(
            torch.Tensor(h, d_model, self.d_v)
        )
        torch.nn.init.ones_(self.W_k)
        torch.nn.init.ones_(self.W_q)
        torch.nn.init.ones_(self.W_v)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k)
        self.linear = torch.nn.Linear(h * self.d_v, d_model)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask_3d: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len = q.size(0), q.size(1)

        q = q.repeat(self.h, 1, 1, 1)
        k = k.repeat(self.h, 1, 1, 1)
        v = v.repeat(self.h, 1, 1, 1)
        q = torch.einsum(
            "hijk,hkl->hijl", (q, self.W_q)
        )
        k = torch.einsum(
            "hijk,hkl->hijl", (k, self.W_k)
        )
        v = torch.einsum(
            "hijk,hkl->hijl", (v, self.W_v)
        )
        q = q.view(self.h * batch_size, seq_len, self.d_k)
        k = k.view(self.h * batch_size, seq_len, self.d_k)
        v = v.view(self.h * batch_size, seq_len, self.d_v)

        if mask_3d is not None:
            mask_3d = mask_3d.view(self.h, 1, 1)

        attention_output = self.scaled_dot_product_attention(
            q, k, v, mask_3d
        )
        attention_output = torch.chunk(attention_output, self.h, dim=0)
        attention_output = torch.cat(attention_output, dim=2)
        output = self.linear(attention_output)
        return output