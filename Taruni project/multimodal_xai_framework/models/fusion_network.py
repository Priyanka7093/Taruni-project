import torch
import torch.nn as nn
from models.image_stream import ImageStreamNetwork
from models.clinical_stream import ClinicalStreamNetwork
from config import CNN_EMBED_DIM, MLP_EMBED_DIM, FUSION_HIDDEN_DIM, NUM_CLASSES

class MultiModalFusionModel(nn.Module):
    """
    Dual-Stream Multi-Modal Fusion Network.
    Concatenates visual imaging embeddings and clinical tabular embeddings
    to perform final diagnostic decision support classification.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(MultiModalFusionModel, self).__init__()
        
        self.image_stream = ImageStreamNetwork(embed_dim=CNN_EMBED_DIM)
        self.clinical_stream = ClinicalStreamNetwork(embed_dim=MLP_EMBED_DIM)
        
        fused_dim = CNN_EMBED_DIM + MLP_EMBED_DIM # 512 + 64 = 576
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(FUSION_HIDDEN_DIM, num_classes)
        )

    def forward(self, image_input, clinical_input):
        img_feats = self.image_stream(image_input)
        clin_feats = self.clinical_stream(clinical_input)
        
        fused_vector = torch.cat([img_feats, clin_feats], dim=1)
        logits = self.fusion_head(fused_vector)
        return logits
