import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, image_embeddings, text_embeddings):
        pass