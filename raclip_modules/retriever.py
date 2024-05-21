import torch
import os
import clip
import csv
from PIL import Image
import numpy as np
import pickle

class Retriever:
    def __init__(self, model_name='ViT-B/32', embeddings_path='./data/reference_embeddings.pkl'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.ref_image_embeddings, self.ref_text_embeddings = self.load_embeddings(embeddings_path)
    
    def load_embeddings(self, embeddings_path):
        """Load embeddings from a pickle file."""
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print("Embeddings have been loaded from a pickle file.")
        
        image_embeddings = torch.stack([x['image_embedding'] for x in embeddings])
        text_embeddings = torch.stack([x['text_embedding'] for x in embeddings])

        return image_embeddings, text_embeddings

    def retrieve_similar(self, image_embeddings, top_k):
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        self.ref_image_embeddings /= self.ref_image_embeddings.norm(dim=-1,keepdim=True)

        similarities = torch.mm(image_embeddings, self.ref_image_embeddings.transpose(0,1))
        top_indices = torch.argsort(similarities)[:,-top_k:]
        
        top_image_embeddings = self.ref_image_embeddings[top_indices]
        top_text_embeddings = self.ref_text_embeddings[top_indices]

        return top_image_embeddings, top_text_embeddings
    
# retriever = Retriever()
# image_path = "./data/reference_set/" + "car.jpg"
# image = Image.open(image_path).convert("RGB")
# image_input = retriever.preprocess(image).unsqueeze(0)
# image_input_1 = retriever.preprocess(image).unsqueeze(0)

# print(retriever.retrieve_similar(retriever.model.encode_image(torch.cat([image_input, image_input_1,image_input_1])), 5))