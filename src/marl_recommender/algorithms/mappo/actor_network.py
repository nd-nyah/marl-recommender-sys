import torch.nn as nn


class ActorNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim=256,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.model(x)