import torch
import pickle
from src.utils import device

class Retriever:
    def __init__(self, embeddings_path):
        self.ref_image_embeddings, self.ref_text_embeddings, self.meta_data = self.load_embeddings(embeddings_path)
        self.ref_image_embeddings = self.ref_image_embeddings.to(device).float()
        self.ref_text_embeddings = self.ref_text_embeddings.to(device).float()
    
    def load_embeddings(self, embeddings_path):
        """Load embeddings from a pickle file."""
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"{len(embeddings)} reference embeddings have been loaded from pickle file: {embeddings_path}")
        
        image_embeddings = torch.stack([x['image_embedding'] for x in embeddings])
        text_embeddings = torch.stack([x['text_embedding'] for x in embeddings])
        meta_data = [x['meta_data'] for x in embeddings]

        return image_embeddings, text_embeddings, meta_data

    def retrieve_similar(self, image_embeddings, top_k):
        image_embeddings_norm = image_embeddings/image_embeddings.norm(dim=-1, keepdim=True)
        ref_image_embeddings_norm = self.ref_image_embeddings/self.ref_image_embeddings.norm(dim=-1,keepdim=True)

        similarities = torch.mm(image_embeddings_norm, ref_image_embeddings_norm.transpose(0,1))
        top_indices = torch.argsort(similarities, descending=True)[:,:top_k]

        top_image_embeddings = self.ref_image_embeddings[top_indices]
        top_text_embeddings = self.ref_text_embeddings[top_indices]

        return top_image_embeddings, top_text_embeddings

    def retrieve_similar_for_image(self, image_embeddings, top_k):
        image_embeddings_norm = image_embeddings/image_embeddings.norm(dim=-1, keepdim=True)
        ref_image_embeddings_norm = self.ref_image_embeddings/self.ref_image_embeddings.norm(dim=-1,keepdim=True)

        similarities = torch.mm(image_embeddings_norm, ref_image_embeddings_norm.transpose(0,1))
        top_indices = torch.argsort(similarities, descending=True)[:,:top_k]

        top_indices = top_indices.squeeze(0)
        
        top_meta_data = [self.meta_data[ind] for ind in top_indices]
        top_image_embeddings = ref_image_embeddings_norm[top_indices]
        top_text_embeddings = self.ref_text_embeddings[top_indices]

        return top_meta_data, top_image_embeddings, top_text_embeddings
    
# retriever = Retriever()
# image_path = "./data/reference_set/" + "car.jpg"
# image = Image.open(image_path).convert("RGB")
# image_input = retriever.preprocess(image).unsqueeze(0)
# image_input_1 = retriever.preprocess(image).unsqueeze(0)

# print(retriever.retrieve_similar(retriever.model.encode_image(torch.cat([image_input, image_input_1,image_input_1])), 5))