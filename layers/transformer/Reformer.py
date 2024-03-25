import torch
from torch.utils.checkpoint import get_device_states, set_device_states

from .Embedding import Embedding
from .FFN import FFN
from .MultiHeadAttention import MultiHeadAttention
from .AddPositionalEncoding import AddPositionalEncoding
from .LshSelfAttetion import LSHSelfAttention



class Deterministic(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)
    
class ReformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, heads_num: int, dropout_rate: float, layer_norm_eps: float) -> None:
        super().__init__()

        self.lshAttention = LSHSelfAttention(d_model, heads_num, bucket_size=32, n_hashes=1)
        self.dropout_self_attention = torch.nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = torch.nn.Dropout(dropout_rate)
        self.layer_norm_ffn = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x1, x2 = x, x
        y1 = x1 + self.lshAttention(self.layer_norm_self_attention(x2))
        y2 = x2 + self.ffn(self.layer_norm_ffn(y1))
        return torch.stack((y1, y2)).mean(dim=0)
    
    def backward(ctx, dy):
        y = ctx.y
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = ctx.layer_norm_ffn(ctx.ffn(y1))
            torch.autograd.backward(gy1, dy2)
        
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = ctx.layer_norm_self_attention(ctx.lshAttention(x2))
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, dx


    def __self_attention_block(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.lshAttention(x)
        return x
    
    def __feed_forward_block(self, x: torch.Tensor) -> torch.Tensor:
        return x
    

class ReformerEncoder(torch.nn.Module):
    def __init__(self, num_embeddings: int, max_len: int, pad_idx: int, d_model: int, N: int, d_ff: int, heads_num: int, dropout_rate: float, layer_norm_eps: float, device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.embedding = Embedding(num_embeddings, d_model, pad_idx)
        self.positional_encoding = AddPositionalEncoding(d_model, max_len, device)
        self.encoder_layers = torch.nn.ModuleList(
            [
                ReformerEncoderLayer(
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


