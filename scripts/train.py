import os
import torch
from torch.utils.data import DataLoader
from src.raclip_modules.model import Model
from src.raclip_modules.retriever import Retriever
from src.raclip_modules.loss import InfoNCELoss
from src.custom_dataset import CustomDataset

TRAIN_SET_FILE = './data/train_set.csv'
MODEL_CHECKPOINT_FOLDER = './.cache/checkpoints'
REFERENCE_EMBEDDINGS_FILE = './.cache/reference_embeddings.pkl'

os.makedirs(MODEL_CHECKPOINT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CustomDataset(TRAIN_SET_FILE)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

retriever = Retriever(REFERENCE_EMBEDDINGS_FILE)
model = Model(retriever)
criterion = InfoNCELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(4):
    for batch in train_loader:
        image_batch = batch['image'].to(device)
        text_batch = batch['text'].to(device)
        
        image_embeddings, text_embeddings = model(image_batch, text_batch)
        loss = criterion(image_embeddings, text_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
    print(f'epoch {epoch + 1} over')

model_file = os.path.join(MODEL_CHECKPOINT_FOLDER, 'model.pth')
torch.save(model.state_dict(), model_file)
print(f'Model saved to {model_file}')