import torch
import os
import clip
from base_vlm import BaseVLM
import csv
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class RaClipVanilla(BaseVLM):
    # Should have all necessary components including clip, reference dataset, k etc..
    def __init__(self, name='RaClip Vanilla', model_name='ViT-B/32', dataset_path='/Users/jalend/Downloads/CSE252D/IMPART/data/reference_set/reference_set.csv'):
        super().__init__(name)
        self.model, self.preprocess = clip.load(model_name, device)
        self.dataset_path = dataset_path
        self.reference_set, self.reference_embeddings = self.load_reference_set()

    # Should load and create the reference set
    def load_reference_set(self):
        """Assuming reference set is a csv like data = [
                ["images/dog1.jpg", "A black Labrador retriever playing in a park."],
                ["images/dog2.jpg", "A small poodle sitting on a sofa."]
            ]
        """
        reference_set = []
        reference_embeddings = []
        dir_path = "/Users/jalend/Downloads/CSE252D/IMPART/data/reference_set/"
        with open(self.dataset_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_path = dir_path + row['image_path']
                description = dir_path + row['description']
                image = Image.open(image_path).convert("RGB")
                processed_image = self.preprocess(image).unsqueeze(0).to(device)
                embedding = self.model.encode_image(processed_image)
                reference_set.append((image_path, description))
                reference_embeddings.append(embedding.squeeze(0))  # Store the embeddings for later retrieval
        
        print(f"Loaded {len(reference_set)} image-text pairs into the reference set.")
        print(reference_embeddings)
        return reference_set, reference_embeddings


    # should implement retrieval from reference set and augmentation  
    def encode_image(self, image_batch):
        image_inputs = torch.stack([self.preprocess(image).unsqueeze(0) for image in image_batch]).to(device)
        image_features = self.model.encode_image(image_inputs)
        return image_features

    # same as clip 
    def encode_text(self, text_batch):
        text_inputs = torch.cat([clip.tokenize(text) for text in text_batch]).to(device)
        text_features = self.model.encode_text(text_inputs)
        return text_features
    
    def retrieve_similar(self, image_batch, top_k=5):
        input_embedding = self.model.encode_image(image_batch)
        similarities = [torch.cosine_similarity(input_embedding, ref_emb, dim=1).item() for ref_emb in self.reference_embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.reference_set[i][0], self.reference_set[i][1], similarities[i]) for i in top_indices]
    
    def augment_image_embedding(self, input_batch, top_k=5):
        input_embedding = self.model.encode_image(input_batch)
        top_similar = self.retrieve_similar(input_batch, top_k)

        # Encode the texts of the top K similar image-text pairs
        texts = [text for _, text, _ in top_similar]
        text_embeddings = self.encode_text(texts)

        combined_embedding = torch.cat((input_embedding, text_embeddings), dim=0)
        augmented_embedding = torch.mean(combined_embedding, dim=0)

        return augmented_embedding
    
model = RaClipVanilla()
model.load_reference_set()