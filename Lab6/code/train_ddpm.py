import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataloader import get_data_loaders
from evaluator import evaluation_model
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import json


class DDPM:
    def __init__(self, config):
        self.config = config
        self.device = config.get(
            "device", "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model = UNet2DModel(
            sample_size=config["image_size"],
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ).to(self.device)

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.config["num_train_timesteps"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config["learning_rate"]
        )

        self.train_loader, self.val_loader = get_data_loaders(
            img_dir=config["img_dir"],
            labels=config["labels"],
            objects=config["objects"],
            batch_size=config["batch_size"],
            train_split=config["train_split"],
        )

    def train(self):
        evaluator = evaluation_model()
        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            progress_bar = tqdm(total=len(self.train_loader))
            progress_bar.set_description(f"Epoch {epoch}")

            for batch in self.train_loader:
                images, labels = batch
                images = images.to(self.device)

                noise = torch.randn(images.shape).to(self.device)
                timesteps = torch.randint(
                    0,
                    self.config["num_train_timesteps"],
                    (images.shape[0],),
                    device=self.device,
                ).long()
                noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

                noise_pred = self.model(noisy_images, timesteps).sample

                loss = nn.MSELoss()(noise_pred, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

            progress_bar.close()

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
                acc = self.evaluate()
                print(f"Epoch {epoch + 1} validation accuracy: {acc}")

    def save_checkpoint(self, epoch):
        pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)
        pipeline.save_pretrained(f"ddpm_model_epoch_{epoch+1}")

    def evaluate(self):
        self.model.eval()
        evaluator = evaluation_model()
        acc = []
        with torch.no_grad():
            for val_batch in self.val_loader:
                val_images, val_labels = val_batch
                val_images = val_images.to(self.device)
                val_labels = val_labels.to(self.device)

                pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)
                generated_images = pipeline(
                    batch_size=val_images.shape[0],
                    num_inference_steps=self.config["num_inference_steps"],
                    output_type="tensor",
                ).images

                acc.append(evaluator.compute_acc(generated_images, val_labels))
        self.model.train()
        return np.mean(acc)

    def test(self):
        self.model.eval()
        evaluator = evaluation_model()

        with open(self.config["test_file"], "r") as f:
            test_data = json.load(f)

        with open(self.config["objects"], "r") as f:
            objects_map = json.load(f)

        for i, test_case in enumerate(test_data):
            label = torch.zeros(24)
            for obj in test_case:
                label[objects_map[obj]] = 1

            pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)
            image = pipeline(
                batch_size=1,
                num_inference_steps=self.config["num_inference_steps"],
                output_type="numpy",
            ).images[0]

            image = (image * 255).round().astype("uint8")
            Image.fromarray(image).save(f"generated_images/test_{i}.png")

            image_tensor = (
                torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )
            image_tensor = image_tensor.to(self.device)
            label = label.unsqueeze(0).to(self.device)

            acc = evaluator.compute_acc(image_tensor, label)
            print(f"Test case {i} accuracy: {acc}")


if __name__ == "__main__":
    config = {
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "num_train_timesteps": 1000,
        "batch_size": 8,
        "image_size": 64,
        "num_inference_steps": 50,
        "train_split": 0.8,
        "img_dir": "dataset/iclevr/",
        "labels": "dataset/train.json",
        "objects": "annotation/objects.json",
    }

    ddpm = DDPM(config)
    ddpm.train()
    ddpm.test()
