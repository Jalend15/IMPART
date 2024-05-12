import clip
import torch
from tqdm import tqdm
from classification.base_classifier import BaseClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClipClassifier(BaseClassifier):

    def __init__(self, name='CLIP'):
        super().__init__(name)
        self.model, self.preprocess = clip.load('ViT-B/32', device)
    
    def get_predictions(self, dataset, classes):
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        test_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
        class_ids = torch.tensor([], device=device)
        predictions = torch.tensor([], device=device)

        for image, class_id in tqdm(test_loader):
            image = image.to(device)
            class_id = class_id.to(device)
            class_ids = torch.cat([class_ids, class_id])

            with torch.no_grad():
                image_features = self.model.encode_image(image)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T

            batch_prediction = torch.argmax(similarity, dim=1)
            predictions = torch.cat([predictions, batch_prediction])
        
        return (predictions, class_ids)