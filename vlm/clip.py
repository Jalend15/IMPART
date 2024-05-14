import clip
import torch
from vlm.base_vlm import BaseVLM

device = "cuda" if torch.cuda.is_available() else "cpu"

class Clip(BaseVLM):
    def __init__(self, name='Clip'):
        super().__init__(name)
        self.model, self.preprocess = clip.load('ViT-B/32', device)
    
    def get_preprocess_transform(self):
        return self.preprocess
    
    def encode_image(self, image_batch):
        image_features = self.model.encode_image(image_batch)
        return image_features

    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(device)
        text_features = self.model.encode_text(text_inputs)
        return text_features