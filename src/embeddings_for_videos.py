import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from vlm.clip_vlm import Clip
import utils
import pickle

# Assuming 'device' and 'clip_preprocess' are properly defined in the 'utils' module.

metadata_path = './data/movies_captions/descriptions.csv'


description_df = pd.read_csv(metadata_path)

description_dict = pd.Series(description_df.description.values, index=description_df.videoid).to_dict()
device = utils.device
print(device)

# Function to load an image and preprocess it
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_input = utils.clip_preprocess(image).unsqueeze(0).to(device)
    return image_input

# Function to get embeddings
def get_embedding(model, image_input):
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features.cpu().numpy()

def extract_video_name(filename):
    # Split by '_' and take the first part, then split by '.' to remove extension details
    return filename.split('_')[0].split('.')[0]

# Path to the directory containing the images
frame_path = './data/movies_captions/2020_embeddings'

# Initialize the model (assuming the model is loaded here, you may need to adjust this)
model = Clip()
# model.to(device)

# Data structure for storing embeddings and descriptions
data = []
data_dict = {}
# Assuming descriptions are in a file or can be inferred from filenames
# If descriptions are in a separate file, load them into a dictionary here

# Process each image in the directory
counter =0 
tmp = set()
for file_name in tqdm(os.listdir(frame_path)):
    if file_name.endswith('.jpg'):  # Check for image files
        full_path = os.path.join(frame_path, file_name)
        image_input = load_image(full_path)
        embedding = get_embedding(model, image_input)
        
        # Assuming description can be derived or is available somehow
        

        video_name = extract_video_name(file_name)
        description = description_dict.get(video_name, 'No description available')
        key = video_name
        if key not in data_dict:
            data_dict[key] = {
                'folder': os.path.basename(full_path),
                'videoname': video_name,
                'embeddings': [],
                'description': description_dict.get(video_name, 'No description available')
            }

        # Append the new embedding to the list of embeddings for this video
        data_dict[key]['embeddings'].append(embedding.tolist())
    # if(len(data_dict)==4):
    #     print(data_dict)
    #     break

# # Convert list to DataFrame
# df = pd.DataFrame(data, columns=['VideoPath','VideoName', 'Embedding', 'Description'])

# Save to CSV
# Path where the pickle file will be stored
pickle_path = './data/movies_captions/2020_embeddings.pkl'

# Serialize the dictionary to a pickle file
with open(pickle_path, 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Data dictionary has been saved to {pickle_path}")
