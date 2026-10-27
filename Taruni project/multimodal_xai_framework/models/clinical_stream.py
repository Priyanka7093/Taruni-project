import torch
import torch.nn as nn
from config import NUM_CLINICAL_FEATURES, MLP_EMBED_DIM

class ClinicalStreamNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for processing clinical patient tabular features.
    Extracts a non-linear feature embedding vector of dimension MLP_EMBED_DIM (64).
    """
    def __init__(self, in_features: int = NUM_CLINICAL_FEATURES, embed_dim: int = MLP_EMBED_DIM):
        super(ClinicalStreamNetwork, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.mlp(x)
