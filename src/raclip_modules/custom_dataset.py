from torch.utils.data import Dataset
import pandas as pd
import clip
from PIL import Image
from src.utils import clip_preprocess

class CustomDataset(Dataset):
    def __init__(self, dataset_file):
        self.dataset = pd.read_csv(dataset_file, keep_default_na=False)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = {}
        try:
            text = self.dataset['description'][idx]
            sample['text'] = clip.tokenize(text, truncate=True).squeeze(0)

            image_path = self.dataset['image_path'][idx]
            image = Image.open(image_path)
            sample['image'] = clip_preprocess(image)
        except Exception as err:
            print(text, self.dataset['image_path'][idx])
            print(f'Got Error {err}')

        return sample