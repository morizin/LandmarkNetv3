import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import timm
from sklearn.cluster import DBSCAN as dbscan
from src.config import hyperparameters
from src.utils import seed_torch
from src.predict import LoFTR, default_cfg

class LandmarkDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image=image)['image']
        return image

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

def main(data_dir, output_dir):
    seed_torch()
    transform = A.Compose([A.Resize(256, 256), A.Normalize()])
    dataset = LandmarkDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=hyperparameters.batch_size, shuffle=True, num_workers=hyperparameters.num_workers)
    
    model = LoFTR(config=default_cfg)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, dataloader, criterion, optimizer, num_epochs=25)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LandmarkNetv3 Training Script')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir)
