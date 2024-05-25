import torch
from tqdm import tqdm
from src.vlm.base_vlm import BaseVLM
from src import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

class ZeroShotClassifier:

    def predict(self, model: BaseVLM, image, classes):
        image_input = utils.clip_preprocess(image).unsqueeze(0).to(device)
        class_texts = [f"a photo of a {c}" for c in classes]
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(class_texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(len(classes))

        for value, index in zip(values, indices):
            print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")

    def evaluate_testset(self, model: BaseVLM, dataset, classes):
        class_texts = [f"a photo of a {c}" for c in classes]
        with torch.no_grad():
            text_features = model.encode_text(class_texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        num_correct = 0
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
        for image_batch, class_id_batch in tqdm(test_loader):
            image_batch = image_batch.to(device)
            class_id_batch = class_id_batch.to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = image_features @ text_features.T
            batch_prediction = torch.argmax(similarity, dim=1)
            num_correct += torch.sum(batch_prediction == class_id_batch).item()
        
        accuracy = num_correct/len(dataset) * 100
        print(f"{model.name} accuracy on {dataset.__class__.__name__}: {accuracy:.2f}")

        return model.name, dataset.__class__.__name__, f'{accuracy:.2f}'