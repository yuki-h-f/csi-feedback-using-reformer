import torch


class FFN(torch.nn.Module):
    """
    Position-wise Feed-Forward Networks
    """
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.nn.functional.relu(self.linear1(x)))