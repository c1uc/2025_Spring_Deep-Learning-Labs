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


def get_data_loaders(
    img_dir, labels, objects, batch_size=64, train_split=0.8, random_seed=42
):
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Create dataset
    dataset = CLEVRDataset(img_dir, labels, objects, transform=transform)

    # Split dataset
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader
