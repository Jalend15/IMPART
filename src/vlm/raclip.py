import torch
import clip
from src.utils import device
from src.vlm.base_vlm import BaseVLM
from src.raclip_modules.model import Model
from src.raclip_modules.retriever import Retriever

class RaClip(BaseVLM):
    def __init__(self, 
                 reference_embeddings_path,
                 model_checkpoint=None,
                 name='RaClip'):
        super().__init__(name)
        self.retriever = Retriever(reference_embeddings_path)
        self.model = Model(self.retriever)
        if model_checkpoint is not None:
            print(f'Loading model from {model_checkpoint}')
            self.model.load_state_dict(torch.load(model_checkpoint))
  
    def encode_image(self, image_batch):
        image_features, _ = self.model(image_batch=image_batch)
        return image_features

    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(device)
        _, text_features = self.model(text_batch=text_inputs)
        return text_features