from vlm.base_vlm import BaseVLM
from raclip.model import Model
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

class RaClip(BaseVLM):
    def __init__(self, retriever, model_checkpoint, name='RaClip'):
        super().__init__(name)
        # load from checkpoint
        self.model = Model(retriever)
  
    def encode_image(self, image_batch):
        image_features, _ = self.model(image_batch=image_batch)
        return image_features

    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(device)
        text_features = self.model(text_batch=text_inputs)
        return text_features