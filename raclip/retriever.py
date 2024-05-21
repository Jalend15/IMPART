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
        self.reference_set = []
        for x in self.reference_embeddings:
            self.reference_set.append([x['image_embedding'],x['text_embedding']])
    def encode_image(self, image_batch):
        image_inputs = torch.stack([self.preprocess(image).unsqueeze(0) for image in image_batch]).to(self.device).float()
        print("Image inputs shape:", image_inputs.shape)  # Debug: Check the input shape
        if image_inputs.dim() == 5:  # Checks if there's an extra batch dimension
            image_inputs = image_inputs.squeeze(1)  # Remove the unnecessary dimension
            print("Image inputs shape:", image_inputs.shape)

        image_features = self.model.encode_image(image_inputs)
        return image_features

    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(self.device)
        text_features = self.model.encode_text(text_inputs)
        return text_features
    # returns k pairs of image, text embeddings from the vector db
    
    def load_embeddings(self, embeddings_path):
        """Load embeddings from a pickle file."""
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print("Embeddings have been loaded from a pickle file.")
        return embeddings

    def retrieve_similar(self, image_batch, top_k=5):
        input_embedding = self.encode_image(image_batch)
        similarities = [torch.cosine_similarity(input_embedding, ref_emb['image_embedding'], dim=1).item() for ref_emb in self.reference_embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.reference_set[i][0], self.reference_set[i][1], similarities[i]) for i in top_indices]
    
retriever = Retriever()
image_path = "./data/reference_set/" + "dog.jpg"
image = Image.open(image_path).convert("RGB")
print(retriever.retrieve_similar(image_batch=[image]))