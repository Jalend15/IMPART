import torch
import clip
from vlm.base_vlm import BaseVLM
from PIL import Image
import raclip_modules.retriever as retriever

device = "cuda" if torch.cuda.is_available() else "cpu"

class RaClipVanilla(BaseVLM):
    # Should have all necessary components including clip, reference dataset, k etc..
    def __init__(self, name='RaClip Vanilla', model_name='ViT-B/32', dataset_path='./data/reference_set/reference_set.csv'):
        super().__init__(name)
        self.model, self.preprocess = clip.load(model_name, device)
        self.dataset_path = dataset_path
        self.retriever = retriever.Retriever()

    # should implement retrieval from reference set and augmentation  
    def encode_image(self, image_batch):
        image_features = self.model.encode_image(image_batch)
        augmented_embedding = self.augment_image_embedding(image_features, 2)
        return augmented_embedding

    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(device)
        text_features = self.model.encode_text(text_inputs)
        return text_features

    def augment_image_embedding(self, input_embedding, top_k=5, weights = [1.0,0.005,0.005]):
        top_image_embeddings, top_text_embeddings = self.retriever.retrieve_similar(input_embedding)
        # Combine embeddings with weights
        weighted_input_embedding = weights[0] * input_embedding
        weighted_image_embeddings = weights[1] * top_image_embeddings
        weighted_text_embeddings = weights[2] * top_text_embeddings

        # Concatenate all embeddings and compute the weighted average
        combined_embedding = torch.cat((weighted_input_embedding, weighted_image_embeddings, weighted_text_embeddings), dim=0)
        augmented_embedding = torch.sum(combined_embedding, dim=0) / sum(weights)

        return augmented_embedding.unsqueeze(0)
    
# model = RaClipVanilla()
# image_path = "./data/reference_set/" + "dog.jpg"
# image = Image.open(image_path).convert("RGB")
# image_input = model.preprocess(image).unsqueeze(0)
# print(model.encode_image(image_input))
