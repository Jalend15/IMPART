import torch
import torch.nn as nn
import clip
from src.utils import device

class Model(nn.Module):
    def __init__(self, retriever, num_attention_layers=2):
        super().__init__()
        self.clip_model, _ = clip.load('ViT-B/32', device)
        self.retriever = retriever
        self.k = 8
        self.crossattn_text_list = nn.ModuleList([nn.MultiheadAttention(512, 8, batch_first=True, device=device) for _ in range(num_attention_layers)])
        self.crossattn_image_list = nn.ModuleList([nn.MultiheadAttention(512, 8, batch_first=True, device=device) for _ in range(num_attention_layers)])

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
    def forward(self, image_batch=None, text_batch=None):
        if image_batch is not None:
            image_embeddings = self.clip_model.encode_image(image_batch).float()

            # (batch_size, k, embd_size), (batch_size, k, embd_size)
            rtvd_image_embeddings, rtvd_text_embeddings = self.retriever.retrieve_similar(image_embeddings, self.k)
            
            for i, (text_attention, image_attention) in enumerate(zip(self.crossattn_text_list, self.crossattn_image_list)):
                if i > 0:
                    image_embeddings = torch.relu(image_embeddings)
                text_augmentation, _ = text_attention(image_embeddings.unsqueeze(1), rtvd_image_embeddings, rtvd_text_embeddings)
                image_augmentation, _ = image_attention(image_embeddings.unsqueeze(1), rtvd_text_embeddings, rtvd_image_embeddings)
                image_embeddings = image_embeddings + text_augmentation.squeeze(1) + image_augmentation.squeeze(1)
        else:
            image_embeddings = None

        if text_batch is not None:
            text_embeddings = self.clip_model.encode_text(text_batch)
        else:
            text_embeddings = None

        return image_embeddings, text_embeddings