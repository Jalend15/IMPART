import torch
import os
import clip
import csv
from PIL import Image
import numpy as np
import pickle

class Retriever:
    # initialize chroma, clip etc..
    def __init__(self, model_name='ViT-B/32', embeddings_path='./data/reference_embeddings.pkl'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.reference_embeddings = self.load_embeddings(embeddings_path)
        self.image_embeddings = torch.stack([x['image_embedding'] for x in self.reference_embeddings])
        self.text_embeddings = torch.stack([x['text_embedding'] for x in self.reference_embeddings])
        self.reference_set = []
        for x in self.reference_embeddings:
            self.reference_set.append([x['image_embedding'],x['text_embedding']])
    
    def load_embeddings(self, embeddings_path):
        """Load embeddings from a pickle file."""
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print("Embeddings have been loaded from a pickle file.")
        return embeddings

    def retrieve_similar(self, image_embeddings, top_k=5):
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        self.image_embeddings /= self.image_embeddings.norm(dim=-1,keepdim=True)
        similarities =torch.stack([torch.mm(image_embeddings, self.image_embeddings.transpose(0,1))])
        top_indices = torch.argsort(similarities).squeeze(0)[:,-top_k:]
        top_embeddings = torch.stack([self.image_embeddings[i] for i in top_indices])
        top_text_embeddings = torch.stack([self.text_embeddings[i] for i in top_indices])

        return top_embeddings, top_text_embeddings
    
# retriever = Retriever()
# image_path = "./data/reference_set/" + "car.jpg"
# image = Image.open(image_path).convert("RGB")
# image_input = retriever.preprocess(image).unsqueeze(0)
# image_input_1 = retriever.preprocess(image).unsqueeze(0)

# print(retriever.retrieve_similar(retriever.model.encode_image(torch.cat([image_input, image_input_1,image_input_1]))))