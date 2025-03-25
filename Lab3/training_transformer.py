import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
import wandb
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime


# TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(
            device=args.device
        )
        self.optim, self.scheduler = self.configure_optimizers()
        self.criterion = nn.CrossEntropyLoss()
        self.prepare_training()

    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        idx = 0
        for data in tqdm(train_loader):
            data = data.to(self.args.device)
            logits, z_idx = self.model(data)

            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), z_idx.reshape(-1))
            total_loss += loss.item()

            idx += 1
            loss.backward()

            if idx % self.args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()

        self.optim.step()
        self.optim.zero_grad()
        self.scheduler.step()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def eval_one_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader):
                data = data.to(self.args.device)
                logits, z_idx = self.model(data)

                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), z_idx.reshape(-1))
                total_loss += loss.item()

            avg_loss = total_loss / len(val_loader)
            return avg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs
        )
        return optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskGIT")
    # TODO2:check your dataset path is correct
    parser.add_argument(
        "--train_d_path",
        type=str,
        default="./dataset/train/",
        help="Training Dataset Path",
    )
    parser.add_argument(
        "--val_d_path",
        type=str,
        default="./dataset/val/",
        help="Validation Dataset Path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints/last_ckpt.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda:1", help="Which device the training is on."
    )
    parser.add_argument("--num_workers", type=int, default=32, help="Number of worker")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--accum-grad", type=int, default=10, help="Number for gradient accumulation."
    )

    # you can modify the hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs to train."
    )
    parser.add_argument(
        "--save-per-epoch",
        type=int,
        default=10,
        help="Save CKPT per ** epochs(default: 1)",
    )
    parser.add_argument(
        "--start-from-epoch", type=int, default=0, help="Number of epochs to train."
    )
    parser.add_argument(
        "--ckpt-interval", type=int, default=0, help="Number of epochs to train."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate."
    )

    parser.add_argument(
        "--MaskGitConfig",
        type=str,
        default="config/MaskGit.yml",
        help="Configurations for TransformerVQGAN",
    )

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, "r"))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )

    wandb.init(project="NYCU_DL_Lab3", name=f"MaskGit_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", config=vars(args))

    # TODO2 step1-5:
    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        train_loss = train_transformer.train_one_epoch(train_loader)
        val_loss = train_transformer.eval_one_epoch(val_loader)

        wandb.log({"train/loss": train_loss, "val/loss": val_loss})
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        
        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.state_dict(), f"transformer_checkpoints/epoch_{epoch}.pt")
            wandb.save(f"transformer_checkpoints/epoch_{epoch}.pt")
