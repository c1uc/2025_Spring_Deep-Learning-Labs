from datetime import datetime
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import (
    Generator,
    Gaussian_Predictor,
    Decoder_Fusion,
    Label_Encoder,
    RGB_Encoder,
)

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

import wandb


def Generate_PSNR(imgs1, imgs2, data_range=1.0):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2)  # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing:
    def __init__(self, args, current_epoch=0):
        # TODO
        self.args = args
        self.current_epoch = current_epoch

        assert args.kl_anneal_type in ["Cyclical", "Monotonic", "Constant"]

        if args.kl_anneal_type == "Cyclical":
            self.betas = self.frange_cycle_linear(
                args.num_epoch,
                n_cycle=args.kl_anneal_cycle,
                ratio=args.kl_anneal_ratio,
            )
        elif args.kl_anneal_type == "Monotonic":
            self.betas = self.frange_cycle_linear(
                args.num_epoch,
                n_cycle=1,
                ratio=args.kl_anneal_ratio,
            )
        elif args.kl_anneal_type == "Constant":
            self.betas = np.ones(args.num_epoch)
        else:
            raise NotImplementedError

    def update(self):
        # TODO
        self.current_epoch += 1

    def get_beta(self):
        # TODO
        return self.betas[self.current_epoch]

    def frange_cycle_linear(self, n_iter, start=0.1, stop=1.0, n_cycle=1, ratio=1):
        # TODO
        betas = np.ones(n_iter) * stop
        period = n_iter // n_cycle

        for c in range(n_cycle):
            start_ = period * c
            end_ = start_ + int(period * ratio)
            betas[start_:end_] = np.linspace(start, stop, int(period * ratio))

        return betas


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(
            args.F_dim + args.L_dim, args.N_dim
        )
        self.Decoder_Fusion = Decoder_Fusion(
            args.F_dim + args.L_dim + args.N_dim, args.D_out_dim
        )

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim = optim.AdamW(self.parameters(), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=args.num_epoch)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

    def forward(self, img, label):
        pass

    def training_stage(self):
        for i in range(1, self.args.num_epoch + 1):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            total_loss = 0

            for img, label in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                total_loss += loss.detach().cpu()

                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar(
                        "train [TeacherForcing: ON, {:.1f}], beta: {:.3f}".format(
                            self.tfr, beta
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )
                else:
                    self.tqdm_bar(
                        "train [TeacherForcing: OFF, {:.1f}], beta: {:.3f}".format(
                            self.tfr, beta
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )

            if i % self.args.per_save == 0:
                self.save(
                    os.path.join(
                        self.args.save_root, f"epoch_{i}.ckpt"
                    )
                )

            wandb.log({
                "train/loss": total_loss / len(train_loader),
                "train/lr": self.scheduler.get_last_lr()[0],
                "train/tfr": self.tfr,
                "train/beta": beta,
            }, step=self.current_epoch)

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        total_loss = 0
        total_psnr = 0
        for img, label in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            self.tqdm_bar(
                "val", pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0]
            )
            
            total_loss += loss.detach().cpu()
            total_psnr += psnr.detach().cpu()

        wandb.log({
            "val/loss": total_loss / len(val_loader),
            "val/lr": self.scheduler.get_last_lr()[0],
            "val/tfr": self.tfr,
            "val/psnr": total_psnr / len(val_loader),
        }, step=self.current_epoch)

    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        total_loss = 0
        beta = self.kl_annealing.get_beta()

        batch_size, timesteps = img.shape[:2]
        x_hat = img[:, 0, ...]

        for t in range(1, timesteps):
            if adapt_TeacherForcing:
                prev = img[:, t - 1, ...]
            else:
                prev = x_hat

            cur = img[:, t, ...]
            cur_label = label[:, t, ...]

            encoded_cur_frame = self.frame_transformation(cur)
            encoded_label = self.label_transformation(cur_label)

            z, mu, logvar = self.Gaussian_Predictor(
                encoded_cur_frame, encoded_label
            )

            encoded_prev_frame = self.frame_transformation(prev).detach()

            decoded = self.Decoder_Fusion(
                encoded_prev_frame, encoded_label, z
            )

            x_hat = self.Generator(decoded)
            x_hat = nn.functional.sigmoid(x_hat)

            mse_loss = self.mse_criterion(x_hat, cur)
            kl_loss = kl_criterion(mu, logvar, batch_size)

            loss = mse_loss + beta * kl_loss
            total_loss += loss


        total_loss /= batch_size

        self.optim.zero_grad()
        total_loss.backward()
        self.optimizer_step()

        return total_loss

    @torch.no_grad()
    def val_one_step(self, img, label):
        # TODO
        total_loss = 0
        total_psnr = 0
        batch_size, timesteps = img.shape[:2]
        x_hat = img[:, 0, ...]

        beta = self.kl_annealing.get_beta()

        for t in range(1, timesteps):
            prev = x_hat
            cur = img[:, t, ...]
            cur_label = label[:, t, ...]
            encoded_cur_frame = self.frame_transformation(cur)
            encoded_label = self.label_transformation(cur_label)

            z, mu, logvar = self.Gaussian_Predictor(
                encoded_cur_frame, encoded_label
            )

            encoded_prev_frame = self.frame_transformation(prev)

            decoded = self.Decoder_Fusion(
                encoded_prev_frame, encoded_label, z
            )

            x_hat = self.Generator(decoded)
            x_hat = nn.functional.sigmoid(x_hat)

            mse_loss = self.mse_criterion(x_hat, cur)
            kl_loss = kl_criterion(mu, logvar, batch_size)

            loss = mse_loss + beta * kl_loss
            total_loss += loss
            total_psnr += Generate_PSNR(x_hat, cur)

        total_loss /= batch_size
        total_psnr /= batch_size * timesteps

        return total_loss, total_psnr

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(
            img_name,
            format="GIF",
            append_images=new_list,
            save_all=True,
            duration=40,
            loop=0,
        )

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )

        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="train",
            video_len=self.train_vi_len,
            partial=args.fast_partial if self.args.fast_train else args.partial,
        )
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )
        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="val",
            video_len=self.val_vi_len,
            partial=1.0,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return val_loader

    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch >= self.args.tfr_sde:
            self.tfr -= self.args.tfr_d_step
            self.tfr = max(self.tfr, 0)

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(
            f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False
        )
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.state_dict(),
                "lr": self.scheduler.get_last_lr()[0],
                "tfr": self.tfr,
                "last_epoch": self.current_epoch,
            },
            path,
        )
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint["state_dict"], strict=True)
            self.args.lr = checkpoint["lr"]
            self.tfr = checkpoint["tfr"]

            self.optim = optim.AdamW(self.parameters(), lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=[20, 40], gamma=0.1
            )
            self.kl_annealing = kl_annealing(
                self.args, current_epoch=checkpoint["last_epoch"]
            )
            self.current_epoch = checkpoint["last_epoch"]

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()


def main(args):

    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003, help="initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument("--gpu", type=str, default="cuda:1")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--store_visualization",
        action="store_true",
        help="If you want to see the result while training",
    )
    parser.add_argument("--DR", type=str, required=True, help="Your Dataset Path")
    parser.add_argument(
        "--save_root", type=str, required=True, help="The path to save your data"
    )
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument(
        "--num_epoch", type=int, default=400, help="number of total epoch"
    )
    parser.add_argument(
        "--per_save", type=int, default=10, help="Save checkpoint every seted epoch"
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Part of the training dataset to be trained",
    )
    parser.add_argument(
        "--train_vi_len", type=int, default=16, help="Training video length"
    )
    parser.add_argument(
        "--val_vi_len", type=int, default=630, help="valdation video length"
    )
    parser.add_argument(
        "--frame_H", type=int, default=32, help="Height input image to be resize"
    )
    parser.add_argument(
        "--frame_W", type=int, default=64, help="Width input image to be resize"
    )

    # Module parameters setting
    parser.add_argument(
        "--F_dim", type=int, default=128, help="Dimension of feature human frame"
    )
    parser.add_argument(
        "--L_dim", type=int, default=32, help="Dimension of feature label frame"
    )
    parser.add_argument("--N_dim", type=int, default=12, help="Dimension of the Noise")
    parser.add_argument(
        "--D_out_dim",
        type=int,
        default=192,
        help="Dimension of the output in Decoder_Fusion",
    )

    # Teacher Forcing strategy
    parser.add_argument(
        "--tfr", type=float, default=1.0, help="The initial teacher forcing ratio"
    )
    parser.add_argument(
        "--tfr_sde",
        type=int,
        default=10,
        help="The epoch that teacher forcing ratio start to decay",
    )
    parser.add_argument(
        "--tfr_d_step",
        type=float,
        default=0.1,
        help="Decay step that teacher forcing ratio adopted",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="The path of your checkpoints"
    )

    # Training Strategy
    parser.add_argument("--fast_train", action="store_true")
    parser.add_argument(
        "--fast_partial",
        type=float,
        default=0.4,
        help="Use part of the training data to fasten the convergence",
    )
    parser.add_argument(
        "--fast_train_epoch",
        type=int,
        default=5,
        help="Number of epoch to use fast train mode",
    )

    # Kl annealing stratedy arguments
    parser.add_argument("--kl_anneal_type", type=str, default="Cyclical", help="")
    parser.add_argument("--kl_anneal_cycle", type=int, default=10, help="")
    parser.add_argument("--kl_anneal_ratio", type=float, default=1, help="")

    args = parser.parse_args()

    wandb.init(project="NYCU_DL_Lab4", name=f"VAE_tfr_{args.tfr}_kl-type_{args.kl_anneal_type}_kl-cycle_{args.kl_anneal_cycle}_kl-ratio_{args.kl_anneal_ratio}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", config=vars(args))

    main(args)
