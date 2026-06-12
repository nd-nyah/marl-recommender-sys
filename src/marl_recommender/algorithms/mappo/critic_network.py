import torch.nn as nn


class CriticNetwork(nn.Module):

    def __init__(
        self,
        state_dim,
        hidden_dim=256,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)