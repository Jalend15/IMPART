import torch
import clip
from PIL import Image
import pickle
from tqdm import tqdm
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

class Loader:
    def __init__(self, model_name='ViT-B/32'):
        self.model, self.preprocess = clip.load(model_name, device)

    def encode_image(self, image):
        image_input = self.preprocess(image).unsqueeze(0).to(device)
        image_features = self.model.encode_image(image_input)
        return image_features

    def encode_text(self, text):
        try:
            # just croping out first 40 words...still does not work for some cases 
            crop_text = " ".join([t for t in text.split()[:40]])
            text_input = clip.tokenize(crop_text).to(device)
            text_features = self.model.encode_text(text_input)
            return text_features
        except Exception as e:
            print(f"Got error while tokenizing: {e}")
            return None

    def load_reference_set(self, reference_set_file):
        """Assuming reference set details is a csv like data = [
                    ["images/dog1.jpg", "A black Labrador retriever playing in a park."],
                    ["images/dog2.jpg", "A small poodle sitting on a sofa."]
                ]
        """
        reference_embeddings = []
        df = pd.read_csv(reference_set_file, keep_default_na=False)        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_path = row['image_path']
            description = row['description']
            image = Image.open(image_path)
            with torch.no_grad():
                image_embedding = self.encode_image(image)
                text_embedding = self.encode_text(description)
            if text_embedding is not None:
                reference_embeddings.append({
                    'meta_data': {
                        'image_path': image_path,
                        'description': description
                    },
                    'image_embedding': image_embedding.squeeze(0),
                    'text_embedding': text_embedding.squeeze(0)
                })
                
        return reference_embeddings

    def save_embeddings(self, embeddings, embedding_file):
        """Save embeddings to a pickle file."""
        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings have been saved to pickle file: {embedding_file}")
        