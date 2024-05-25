from torch.utils.data import Dataset
import pandas as pd
import clip
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, dataset_file):
        self.dataset = pd.read_csv(dataset_file, keep_default_na=False)
        _, self.preprocess = clip.load('ViT-B/32')
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = {}
        try:
            text = self.dataset['description'][idx]
            crop_text = " ".join([t for t in text.split()[:40]])
            sample['text'] = clip.tokenize(crop_text).squeeze(0)

            image_path = self.dataset['image_path'][idx]
            image = Image.open(image_path)
            sample['image'] = self.preprocess(image)
        except Exception as err:
            print(text, self.dataset['image_path'][idx])
            print(f'Got Error {err}')

        return sample