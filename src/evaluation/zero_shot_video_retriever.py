import pickle
import torch
import clip
from src.utils import device
from src import utils

class ZeroShotVideoRetriever:
    def __init__(self, video_summary_file='../.cache/video_summary.pkl'):
        self.model, self.preprocess = clip.load('ViT-B/32', device)
        self.video_names, self.descriptions, self.movie_names, self.video_embeddings = self.load_video_summary(video_summary_file)


    def load_video_summary(self, embeddings_path):
        """Load video embeddings from a pickle file."""
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"{len(embeddings)} videos embeddings have been loaded from pickle file: {embeddings_path}")
        
        video_names = []
        descriptions = []
        movie_names = []
        video_embeddings = []
        for key in embeddings:
            video_names.append(embeddings[key]['videoname'])
            descriptions.append(embeddings[key]['description'])
            movie_names.append(embeddings[key]['movie_name'])
            video_embedding = torch.tensor(embeddings[key]['embeddings']).squeeze(1)
            video_embeddings.append(video_embedding.mean(0))
        video_embeddings = torch.stack(video_embeddings).to(device).float()

        return video_names, descriptions, movie_names, video_embeddings

    def retrieve_video(self, text_prompt, top_k):
        text_input = clip.tokenize(text_prompt).to(device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_input)
        text_embeddings_norm = text_embeddings/text_embeddings.norm(dim=-1, keepdim=True)
        print(text_embeddings_norm.shape)
        
        video_embeddings_norm = self.video_embeddings/self.video_embeddings.norm(dim=-1,keepdim=True)

        similarities = torch.mm(text_embeddings_norm, video_embeddings_norm.transpose(0,1))
        top_indices = torch.argsort(similarities, descending=True)[:,:top_k]

        top_indices = top_indices.squeeze(0)
        
        top_video_names = [self.video_names[ind] for ind in top_indices]
        top_descriptions = [self.descriptions[ind] for ind in top_indices]
        top_movie_names = [self.movie_names[ind] for ind in top_indices]

        return top_video_names, top_descriptions, top_movie_names
    
    def retrieve_video_from_image(self, image, top_k):
        image_input = utils.clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embeddings = self.model.encode_image(image_input)
        image_embeddings_norm = image_embeddings/image_embeddings.norm(dim=-1, keepdim=True)
        print(image_embeddings_norm.shape)
        
        video_embeddings_norm = self.video_embeddings/self.video_embeddings.norm(dim=-1,keepdim=True)

        similarities = torch.mm(image_embeddings_norm, video_embeddings_norm.transpose(0,1))
        top_indices = torch.argsort(similarities, descending=True)[:,:top_k]

        top_indices = top_indices.squeeze(0)
        
        top_video_names = [self.video_names[ind] for ind in top_indices]
        top_descriptions = [self.descriptions[ind] for ind in top_indices]
        top_movie_names = [self.movie_names[ind] for ind in top_indices]

        return top_video_names, top_descriptions, top_movie_names

    def run():
        pass