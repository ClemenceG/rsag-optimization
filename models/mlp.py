import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module): 
    """
    Very simple
    2 hidden layer MLP with 512 and 512 hidden units respectively
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # x = x.view(-1,28*28)
        # x = F.relu(self.layers[0](x))
        return self.layers(x)