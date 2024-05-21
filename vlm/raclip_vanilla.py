from vlm.base_vlm import BaseVLM
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class RaClipVanilla(BaseVLM):
    def __init__(self, retriever, name='RaClip Vanilla'):
        super().__init__(name)
        self.retriever = retriever
        self.k = 16

    def encode_image(self, image_batch):
        image_embeddings = self.clip_model.encode_image(image_batch)

        # (batch_size, k, embd_size), (batch_size, k, embd_size)
        rtvd_image_embeddings, rtvd_text_embeddings = self.retriever.retrieve(image_embeddings, self.k)
        # image_embeddings = RAM Logic

        return image_embeddings

    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(device)
        text_features = self.model.encode_text(text_inputs)
        return text_features