import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random


class CLEVRDataset(Dataset):
    def __init__(self, img_dir, labels, objects, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Load annotations
        with open(labels, "r") as f:
            self.annotations = json.load(f)

        with open(objects, "r") as f:
            self.label_to_idx = json.load(f)

        self.img_files = list(self.annotations.keys())

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Create label tensor
        labels = self.annotations[img_file]
        label_tensor = torch.zeros(len(self.label_to_idx))
        for label in labels:
            label_tensor[self.label_to_idx[label]] = 1

        return image, label_tensor

def get_data_loader(img_dir, labels, objects, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = CLEVRDataset(img_dir, labels, objects, transform=transform)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    return train_loader
