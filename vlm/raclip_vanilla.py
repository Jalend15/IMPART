from vlm.base_vlm import BaseVLM
import torch
import os
import clip
import csv
import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

class RaClipVanilla(BaseVLM):
    # Should have all necessary components including clip, reference dataset, k etc..
    def __init__(self, name='RaClip Vanilla', model_name='ViT-B/32', dataset_path='path/to/dataset'):
        super().__init__(name)
        self.model, self.preprocess = clip.load(model_name, device)
        self.dataset_path = dataset_path
        self.reference_set = []
        self.reference_embeddings = []

    # Should load and create the reference set
    def load_reference_set(self, dataset):
        """Assuming reference set is a csv like data = [
                ["images/dog1.jpg", "A black Labrador retriever playing in a park."],
                ["images/dog2.jpg", "A small poodle sitting on a sofa."]
            ]
        """
        with open(self.dataset_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_path = row['image_path']
                description = row['description']
                image = Image.open(image_path).convert("RGB")
                processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(processed_image)
                self.reference_set.append((image_path, description))
                self.reference_embeddings.append(embedding.squeeze(0))  # Store the embeddings for later retrieval

        print(f"Loaded {len(self.reference_set)} image-text pairs into the reference set.")


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