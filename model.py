import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg19(weights=VGG19_Weights).features[:29]
        self.layers = [0, 5, 10, 19, 28]

    def forward(self, x):
        features = []
        for num_layer, layer in enumerate(self.model):
            x = layer(x)
            if num_layer in self.layers:
                features.append(x)
        return features

        