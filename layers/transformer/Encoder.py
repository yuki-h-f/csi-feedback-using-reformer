import torch

from .Embedding import Embedding
from .FFN import FFN
from .MultiHeadAttention import MultiHeadAttention
from .AddPositionalEncoding import AddPositionalEncoding

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, heads_num: int, dropout_rate: float, layer_norm_eps: float) -> None:
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, heads_num)
        self.dropout_self_attention = torch.nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = torch.nn.Dropout(dropout_rate)
        self.layer_norm_ffn = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x1 = self.__self_attention_block(self.layer_norm_self_attention(x), mask)
        x2 = self.__feed_forward_block((self.layer_norm_ffn(x1 + x)))
        return x + x1 + x2

    def __self_attention_block(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.multi_head_attention(x, x, x, mask)
        return x
    
    def __feed_forward_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
    

class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_embeddings: int, max_len: int, pad_idx: int, d_model: int, N: int, d_ff: int, heads_num: int, dropout_rate: float, layer_norm_eps: float, device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.embedding = Embedding(num_embeddings, d_model, pad_idx)
        self.positional_encoding = AddPositionalEncoding(d_model, max_len, device)
        self.encoder_layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, d_ff, heads_num, dropout_rate, layer_norm_eps
                ) for _ in range(N)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear = torch.nn.Linear(d_model, 32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x

