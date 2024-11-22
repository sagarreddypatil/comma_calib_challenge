import torch
from torch import nn

from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights

class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        weights = RegNet_X_400MF_Weights if pretrained else None
        self.model = regnet_x_400mf(weights=weights)

    def forward(self, x):
        return self.model(x)
    

if __name__ == "__main__":
    sample_input = torch.randn(1, 1, 120, 90)

