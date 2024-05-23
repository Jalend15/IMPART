import requests
import os
import pandas as pd
from bs4 import BeautifulSoup
import urllib.parse
from tqdm import tqdm

def clean_description(description_with_html):
    # Decode URL-encoded characters
    description_with_html = description_with_html.replace('+', ' ')
    decoded_description = urllib.parse.unquote(description_with_html)
    
    # Parse HTML and remove tags
    soup = BeautifulSoup(decoded_description, 'html.parser')
    clean_description = soup.get_text(separator=' ')
    
    # Remove extra whitespace and long words
    final_text_list = []
    for text in clean_description.split():
        if len(text) <= 25:
            final_text_list.append(text)
    clean_description = ' '.join(final_text_list)
    
    return clean_description

def download_image(url, path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return True
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return False
    except Exception as err:
        print(f"An error occurred: {err}")
        return False


YFCC_FILE = './data/yfcc1k.csv'
IMAGE_FOLDER = './data/images'
OUT_FILE = './data/reference_set_details.csv'

df = pd.read_csv(YFCC_FILE)

image_data = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    photo_url = row['downloadurl']
    description = row['description']
    description = clean_description(description)
    
    # Generate a safe filename from the index
    filename = f"{index}.jpg" 
    image_path = os.path.join(IMAGE_FOLDER, filename)
    # Download the image
    if download_image(photo_url, image_path):
        image_data.append(
            {
                'image_path' : image_path.replace('\\','/'),
                'description': description
            }
        )

image_df = pd.DataFrame(image_data)
image_df.to_csv(OUT_FILE, index=False)