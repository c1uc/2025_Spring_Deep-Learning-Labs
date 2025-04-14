from Tester import Test_model
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from modules import (
    Generator,
    Gaussian_Predictor,
    Decoder_Fusion,
    Label_Encoder,
    RGB_Encoder,
)
from torchvision.utils import save_image
from torch import stack

import imageio
from math import log10
import glob
import pandas as pd
import matplotlib.pyplot as plt
from Trainer import Generate_PSNR

class Plot(Test_model):
    def __init__(self, args):
        super(Plot, self).__init__(args)

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for idx, (img, label) in enumerate(tqdm(val_loader, ncols=80)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            self.val_one_step(img, label, idx)

    def val_one_step(self, img, label, idx=0):
        img = img.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)
        assert label.shape[0] == 630, "Testing pose seqence should be 630"
        # assert img.shape[0] == 1, "Testing video seqence should be 1"

        # decoded_frame_list is used to store the predicted frame seq
        # label_list is used to store the label seq
        # Both list will be used to make gif
        decoded_frame_list = [img[0].cpu()]
        label_list = []

        # TODO
        for t in range(1, label.shape[0]):
            prev = decoded_frame_list[-1].to(self.args.device)
            lbl = label[t, ...].to(self.args.device)

            encoded_img = self.frame_transformation(prev)
            encoded_label = self.label_transformation(lbl)

            z, mu, logvar = self.Gaussian_Predictor(encoded_img, encoded_label)
            eps = torch.randn_like(z)

            decoded_frame = self.Decoder_Fusion(encoded_img, encoded_label, eps)
            
            x_hat = self.Generator(decoded_frame)
            x_hat = nn.functional.sigmoid(x_hat)

            decoded_frame_list.append(x_hat.cpu())
            label_list.append(lbl.cpu())
            
            
        # Please do not modify this part, it is used for visulization
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)

        assert generated_frame.shape == (
            1,
            630,
            3,
            32,
            64,
        ), f"The shape of output should be (1, 630, 3, 32, 64), but your output shape is {generated_frame.shape}"

        groundtruth = img.permute(1, 0, 2, 3, 4)[0].to(self.args.device)
        generated = generated_frame[0].to(self.args.device)
        psnr_list = []

        for i in range(1, 630):
            psnr = Generate_PSNR(groundtruth[i], generated[i])
            psnr_list.append(psnr.cpu().numpy())

        x = np.arange(1, 630)
        plt.plot(x, psnr_list, label="PSNR per frame")
        plt.title("PSNR per frame, average: " + str(np.mean(psnr_list)))
        plt.savefig(os.path.join(self.args.save_root, f"psnr.png"))
        plt.close()

def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = Plot(args).to(args.device)
    model.load_checkpoint()
    model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no_sanity", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--make_gif", action="store_true")
    parser.add_argument("--DR", type=str, required=True, help="Your Dataset Path")
    parser.add_argument(
        "--save_root", type=str, required=True, help="The path to save your data"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_epoch", type=int, default=70, help="number of total epoch"
    )
    parser.add_argument(
        "--per_save", type=int, default=3, help="Save checkpoint every seted epoch"
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

    main(args)
