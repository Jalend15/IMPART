import os
import torch
import time
from torch.utils.data import DataLoader
from src.utils import device
from src.raclip_modules.model import Model
from src.raclip_modules.retriever import Retriever
from src.raclip_modules.loss import InfoNCELoss
from IMPART.src.raclip_modules.custom_dataset import CustomDataset

TRAIN_SET_FILE = './data/train_set_10k.csv'
MODEL_CHECKPOINT_FOLDER = './.cache/checkpoints/10k'
REFERENCE_EMBEDDINGS_FILE = './.cache/reference_embeddings_1k.pkl'
BATCH_SIZE=100

os.makedirs(MODEL_CHECKPOINT_FOLDER, exist_ok=True)
print(f'Device being used: {device}')

dataset = CustomDataset(TRAIN_SET_FILE)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

retriever = Retriever(REFERENCE_EMBEDDINGS_FILE)
model = Model(retriever)
criterion = InfoNCELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

model_file = os.path.join(MODEL_CHECKPOINT_FOLDER, f'model_0.pth')
torch.save(model.state_dict(), model_file)
print(f'Initial model saved to {model_file}')

for epoch in range(8):
    end = 0
    for i, batch in enumerate(train_loader):
        image_batch = batch['image'].to(device)
        text_batch = batch['text'].to(device)
        
        image_embeddings, text_embeddings = model(image_batch, text_batch)
        loss = criterion(image_embeddings, text_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%10 == 9:
            print(f'Images Seen: {(i+1)*BATCH_SIZE}, Loss: {loss}')
    model_file = os.path.join(MODEL_CHECKPOINT_FOLDER, f'model_{epoch+1}.pth')
    torch.save(model.state_dict(), model_file)
    print(f'Model saved to {model_file}')
    print(f'epoch {epoch + 1} over')