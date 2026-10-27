import torch
import torch.nn as nn
import torchvision.models as models
from config import CNN_EMBED_DIM

class ImageStreamNetwork(nn.Module):
    """
    CNN Backbone (ResNet18) for processing medical brain MRI images.
    Extracts a high-level visual embedding vector of dimension CNN_EMBED_DIM (512).
    """
    def __init__(self, embed_dim: int = CNN_EMBED_DIM, pretrained: bool = True):
        super(ImageStreamNetwork, self).__init__()
        
        # Load ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Save reference to final convolutional layer for Grad-CAM visual extraction
        self.target_conv_layer = self.backbone.layer4[1].conv2

        # Replace classification FC head with an identity / feature projection layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        return self.target_conv_layer
