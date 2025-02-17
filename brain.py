import torch
import torch.nn as nn


class Brain(nn.Module):
    def __init__(self, inputsize):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputsize, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2),
            nn.Sigmoid()
        )

    def decide(self, surrounding):
        if not isinstance(surrounding, torch.Tensor):
            surrounding = torch.tensor(surrounding).to(torch.float32)
        return self.model(surrounding)

