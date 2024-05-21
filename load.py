# script to load reference set, embed using clip and store in vector db (may be chroma)
import torch
import os
import clip
import csv
from PIL import Image
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"

class Load():
    # Should have all necessary components including clip, reference dataset, k etc..
    def __init__(self, model_name='ViT-B/32', dataset_path='./data/reference_set/reference_set.csv'):
        self.model, self.preprocess = clip.load(model_name, device)
        self.dataset_path = dataset_path
        self.reference_set, self.reference_embeddings = self.load_reference_set()

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

    def load_reference_set(self):
        """Assuming reference set is a csv like data = [
                    ["images/dog1.jpg", "A black Labrador retriever playing in a park."],
                    ["images/dog2.jpg", "A small poodle sitting on a sofa."]
                ]
        """
        reference_set = []
        reference_embeddings = []
        torch.no_grad()
        dir_path = "./data/reference_set/"
        with open(self.dataset_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_path = dir_path + row['image_path']
                description = dir_path + row['description']
                image = Image.open(image_path).convert("RGB")
                processed_image = self.preprocess(image).unsqueeze(0).to(device)
                image_embedding = self.encode_image([image])
                text_embedding = self.encode_text([description])
                reference_set.append((image_path, description))
                reference_embeddings.append({
                    'image_embedding': image_embedding.squeeze(0),
                    'text_embedding': text_embedding.squeeze(0)
                })

        self.save_embeddings(reference_embeddings)
        return reference_set, reference_embeddings

    def save_embeddings(self, embeddings):
        """Save embeddings to a pickle file."""
        with open('./data/reference_embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        print("Embeddings have been saved to a pickle file.")
        
Load()