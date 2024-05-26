import torch
import clip
from src.utils import device
from src.vlm.base_vlm import BaseVLM
from src.raclip_modules.model import Model
from src.raclip_modules.retriever import Retriever

class RaClip(BaseVLM):
    def __init__(self, 
                 model_checkpoint='../.cache/checkpoints/model.pth',
                 reference_embeddings_path = '../.cache/reference_embeddings.pkl',
                 name='RaClip'):
        super().__init__(name)
        self.retriever = Retriever(reference_embeddings_path)
        self.model = Model(self.retriever)
        self.model.load_state_dict(torch.load(model_checkpoint))
  
    def encode_image(self, image_batch):
        image_features, _ = self.model(image_batch=image_batch)
        return image_features

    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(device)
        _, text_features = self.model(text_batch=text_inputs)
        return text_features