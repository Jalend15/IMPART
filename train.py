import torch
from raclip.model import Model
from raclip.loss import InfoNCELoss
from raclip.retriever import Retriever

device = "cuda" if torch.cuda.is_available() else "cpu"

# load training dataset
train_loader = None

retriever = Retriever()
model = Model(retriever)

criterion = InfoNCELoss()
# need to filter trainable params
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(4):
    for image_batch, text_batch in train_loader:
        image_batch = image_batch.to(device)
        label_batch = text_batch.to(device)
        
        image_embeddings, text_embeddings = model(image_batch, text_batch)
        loss = criterion(image_embeddings, text_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()