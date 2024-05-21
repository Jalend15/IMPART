import torch
import os
import clip
from base_vlm import BaseVLM
import csv
from PIL import Image
import numpy as np
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
        image_inputs = torch.stack([self.preprocess(image).unsqueeze(0) for image in image_batch]).to(device).float()
        print("Image inputs shape:", image_inputs.shape)  # Debug: Check the input shape
        if image_inputs.dim() == 5:  # Checks if there's an extra batch dimension
            image_inputs = image_inputs.squeeze(1)  # Remove the unnecessary dimension
            print("Image inputs shape:", image_inputs.shape)

        image_features = self.model.encode_image(image_inputs)
        return image_features

    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(device)
        text_features = self.model.encode_text(text_inputs)
        return text_features

    def augment_image_embedding(self, input_image_paths, top_k=5):
        input_batch = [Image.open(image_path).convert("RGB") for image_path in input_image_paths]
        input_embedding = self.encode_image(input_batch)
        top_image_embeddings, top_text_embeddings = self.retriever.retrieve_similar(image_batch=input_batch)
        combined_embedding = torch.cat((input_embedding, top_image_embeddings), dim=0)
        augmented_embedding = torch.mean(combined_embedding, dim=0)

        return augmented_embedding
    
model = RaClipVanilla()
image_path = "./data/reference_set/" + "dog.jpg"
image = Image.open(image_path).convert("RGB")
print(model.augment_image_embedding([image_path],2))
