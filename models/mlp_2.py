import torch
from torch import nn
from torch.nn import functional as F

class Two_Layer_MLP(nn.Module): 
    """
    Very simple
    2 hidden layer MLP with 512 and 512 hidden units respectively
    """
    def __init__(self, input_dim=28*28, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # x = x.view(-1,28*28)
        # x = F.relu(self.layers[0](x))
        return self.layers(x)