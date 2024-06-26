import torch
import clip
from src.utils import device
from src.vlm.base_vlm import BaseVLM
from src.raclip_modules.retriever import Retriever

class RaClipVanilla(BaseVLM):
    def __init__(self, 
                 reference_embeddings_path,
                 name='RaClip Vanilla', 
                 model_name='ViT-B/32'):
        super().__init__(name)
        self.model, self.preprocess = clip.load(model_name, device)
        self.retriever = Retriever(reference_embeddings_path)
 
    def encode_image(self, image_batch):
        image_embeddings = self.model.encode_image(image_batch)
        top_image_embeddings, top_text_embeddings = self.retriever.retrieve_similar(image_embeddings, 8)
        augmented_embedding = self.augment_image_embedding(image_embeddings, top_image_embeddings, top_text_embeddings)
        return augmented_embedding

    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(device)
        text_features = self.model.encode_text(text_inputs)
        return text_features

    def augment_image_embedding(self, input_embedding, top_image_embeddings, top_text_embeddings, weights = [1,0.5,0.05]):
        # Combine embeddings with weights
        weighted_input_embedding = weights[0] * input_embedding
        weighted_image_embeddings = weights[1] * top_image_embeddings / top_image_embeddings.shape[1]
        weighted_text_embeddings = weights[2] * top_text_embeddings / top_text_embeddings.shape[1]

        # Concatenate all embeddings and compute the weighted average
        combined_embedding = torch.cat((weighted_input_embedding.unsqueeze(1), weighted_image_embeddings, weighted_text_embeddings), dim=1)
        augmented_embedding = torch.sum(combined_embedding, dim=1) / sum(weights)

        return augmented_embedding
    

# model = RaClipVanilla()
# image_path = "./data/reference_set/" + "dog.jpg"
# image = Image.open(image_path).convert("RGB")
# image_input = model.preprocess(image).unsqueeze(0)
# image_input_1 = model.preprocess(image).unsqueeze(0)
# image_input_2 = model.preprocess(image).unsqueeze(0)

# # print(model.encode_image(torch.cat([image_input, image_input_1,image_input_2])))
# print(model.encode_image(torch.cat([image_input])))
