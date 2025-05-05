import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataloader import get_data_loader
from evaluator import evaluation_model
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import json
from torchvision.utils import make_grid, save_image

class DDPM_UNet(nn.Module):
    def __init__(self, config):
        super(DDPM_UNet, self).__init__()
        
        self.device = config.get(
            "device", "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        self.model = UNet2DModel(
            sample_size=config["image_size"],
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            class_embed_type="identity",
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
        
        self.class_embedding = nn.Linear(24, 512).to(self.device)
        
    def forward(self, x, t, y):
        class_emb = self.class_embedding(y)
        x = self.model(x, t, class_emb)
        return x

class DDPM:
    def __init__(self, config):
        self.config = config
        self.device = config.get(
            "device", "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model = DDPM_UNet(config)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config["num_train_timesteps"],
            beta_schedule=self.config["beta_schedule"],
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config["learning_rate"]
        )

        self.train_loader = get_data_loader(
            img_dir=config["img_dir"],
            labels=config["labels"],
            objects=config["objects"],
            batch_size=config["batch_size"],
        )

        self.evaluator = evaluation_model(self.device)

    def train(self):
        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            progress_bar = tqdm(total=len(self.train_loader))
            progress_bar.set_description(f"Epoch {epoch}")

            for batch in self.train_loader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                noise = torch.randn(images.shape).to(self.device)
                timesteps = torch.randint(
                    0,
                    self.config["num_train_timesteps"],
                    (images.shape[0],),
                    device=self.device,
                ).long()
                noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

                noise_pred = self.model(noisy_images, timesteps, labels).sample

                loss = nn.MSELoss()(noise_pred, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

            progress_bar.close()

            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        path = f"checkpoint_{self.config['num_train_timesteps']}_steps/ddpm_model_epoch_{epoch+1}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(data, path)
        
    def load_checkpoint(self, path):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])

    def test(self, test_file):
        self.model.eval()

        with open(test_file, "r") as f:
            test_data = json.load(f)

        with open(self.config["objects"], "r") as f:
            objects_map = json.load(f)
            
        test_dir = f"images/{test_file.split('/')[-1].split('.')[0]}"
        os.makedirs(test_dir, exist_ok=True)
        
        results = []
        accs = []
        
        for idx, test_case in enumerate(test_data):
            label = torch.zeros(24)
            for obj in test_case:
                label[objects_map[obj]] = 1
                
            denoise_process = []

            x = torch.randn(1, 3, 64, 64).to(self.device)
            y = label.unsqueeze(0).to(self.device)
            
            for i, t in enumerate(self.noise_scheduler.timesteps):
                with torch.no_grad():
                    r = self.model(x, t, y).sample

                x = self.noise_scheduler.step(r, t, x).prev_sample
                
                if i % (self.config["num_inference_steps"] // 10) == 0:
                    denoise_process.append(x.clone())
                
            acc = self.evaluator.eval(x, y)
            accs.append(acc)
            
            denoise_process.append(x.clone())
            denoise_process = torch.cat(denoise_process, dim=0)
            
            image_tensor = make_grid(denoise_process / 2 + 0.5, nrow=denoise_process.shape[0])
            save_image(image_tensor, f"{test_dir}/process_{idx}.png")
            
            save_image(x / 2 + 0.5, f"{test_dir}/{idx}.png")
            
            results.append(x)
            
        results = torch.cat(results, dim=0)
        results = make_grid(results / 2 + 0.5, nrow=8)
        save_image(results, f"{test_dir}/results.png")
        
        print(f"Average accuracy of {test_file}: {sum(accs) / len(accs)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=100)
    args = parser.parse_args()
    
    timesteps = args.timesteps
    config = {
        "device": "cuda:1",
        "learning_rate": 1e-5,
        "num_epochs": 200,
        "batch_size": 32,
        "image_size": 64,
        "num_train_timesteps": timesteps,
        "num_inference_steps": timesteps,
        "beta_schedule": "squaredcos_cap_v2",
        "img_dir": "iclevr/",
        "labels": "annotation/train.json",
        "objects": "annotation/objects.json",
    }

    ddpm = DDPM(config)
    if args.test is None:
        ddpm.train()
        
    ddpm.load_checkpoint(f"checkpoint_{timesteps}_steps/ddpm_model_epoch_250.pth" if args.test is None else args.test)
    for f in ["annotation/test.json", "annotation/new_test.json", "annotation/extra.json"]:
        ddpm.test(f)
