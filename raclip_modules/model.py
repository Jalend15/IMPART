import torch.nn as nn
import clip

class Model(nn.Module):
    def __init__(self, retriever):
        super().__init__()
        self.clip_model, self.preprocess = clip.load('ViT-B/32')
        self.retriever = retriever
        self.k = 16
        self.multihead_cross_attn_text = nn.MultiheadAttention(512, 8, batch_first=True)
        self.multihead_cross_attn_image = nn.MultiheadAttention(512, 8, batch_first=True)

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
    def forward(self, image_batch=None, text_batch=None):
        if image_batch is not None:
            image_embeddings = self.clip_model.encode_image(image_batch)

            # (batch_size, k, embd_size), (batch_size, k, embd_size)
            rtvd_image_embeddings, rtvd_text_embeddings = self.retriever.retrieve(image_embeddings, self.k)
            
            text_augmentation, _ = self.multihead_cross_attn_text(image_embeddings.unsqueeze(1), rtvd_image_embeddings, rtvd_text_embeddings)
            image_augmentation, _ = self.multihead_cross_attn_text(image_embeddings.unsqueeze(1), rtvd_text_embeddings, rtvd_image_embeddings)
            image_embeddings = image_embeddings + text_augmentation + image_augmentation
        else:
            image_embeddings = None

        if text_batch is not None:
            text_embeddings = self.clip_model.encode_text(text_batch)
        else:
            text_embeddings = None

        return image_embeddings, text_embeddings