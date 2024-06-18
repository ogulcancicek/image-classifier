import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_model import ExampleDataset
from model import ExampleModel

train_folder = './model_dataset/train'
test_folder = './model_dataset/test'
valid_folder = './model_dataset/test'

num_classes = 4
batch_size = 8
num_epochs = 8
train_losses, val_losses = [], []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]
)

# Dataset
train_dataset = ExampleDataset(train_folder, transform=transform)
test_dataset = ExampleDataset(test_folder, transform=transform)
valid_dataset = ExampleDataset(valid_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# Model
model = ExampleModel(num_classes=num_classes)
model.to(device)

# Params
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training loop"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() + labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc='Validation loop'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() + labels.size(0)

        val_loss = running_loss / len(train_loader.dataset)
        val_losses.append(val_loss)

    print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss}, Validation Loss: {val_loss}')

    os.makedirs('model', exist_ok=True)
    torch.save(obj=model.state_dict(), f=f'model/animal_{epoch}.pth')

