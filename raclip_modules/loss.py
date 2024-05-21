import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_embeddings, text_embeddings):
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

        similarities = (image_embeddings @ text_embeddings.T)/self.temperature

        x = similarities.softmax(0)
        x = -torch.log(x)
        loss = x.trace()/x.shape[0]

        x = similarities.softmax(1)
        x = -torch.log(x)
        loss += x.trace()/x.shape[0]

        return loss