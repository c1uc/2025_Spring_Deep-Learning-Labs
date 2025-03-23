import argparse
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from oxford_pet import load_dataset
from models import UNet, ResNet34Unet
from utils import dice_loss, dice_score
from evaluate import evaluate
from tqdm import tqdm
import wandb


def ensure_data_dir(data_path):
    import os

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.listdir(data_path):
        from oxford_pet import OxfordPetDataset

        OxfordPetDataset.download(data_path)


def train(args):
    # implement the training function here
    ensure_data_dir(args.data_path)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    if args.model == "unet":
        model = UNet()
    elif args.model == "resnet34_unet":
        model = ResNet34Unet()
    else:
        raise ValueError(f"Model {args.model} not found")

    model.to(device)

    train_dataset = load_dataset(args.data_path, "train")
    valid_dataset = load_dataset(args.data_path, "valid")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = torch.nn.BCELoss()
    best_eval_dice_score = 0.90

    for epoch in range(args.epochs):
        model.train()
        train_dice_score, train_bce_loss, train_dice_loss = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = batch["image"].to(device).float()
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            pred_masks = model(images)

            d_score = dice_score(pred_masks, masks)
            b_loss = loss_fn(pred_masks, masks)
            d_loss = dice_loss(pred_masks, masks)

            loss = b_loss + d_loss
            loss.backward()
            optimizer.step()

            train_dice_score += d_score.item()
            train_bce_loss += b_loss.item()
            train_dice_loss += d_loss.item()

        eval_dice_score, eval_bce_loss, eval_dice_loss = evaluate(
            model, valid_loader, device
        )
        if eval_dice_score > best_eval_dice_score:
            best_eval_dice_score = eval_dice_score
            torch.save(
                model.state_dict(),
                f"saved_models/{args.model}_{epoch}_{eval_dice_score:.4f}.pth",
            )

        if args.wandb:
            wandb.log(
                {
                    "train/dice_score": train_dice_score / len(train_loader),
                    "train/bce_loss": train_bce_loss / len(train_loader),
                    "train/dice_loss": train_dice_loss / len(train_loader),
                    "valid/dice_score": eval_dice_score,
                    "valid/bce_loss": eval_bce_loss,
                    "valid/dice_loss": eval_dice_loss,
                }
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs} - Train Dice Score: {train_dice_score / len(train_loader):.4f} - Valid Dice Score: {eval_dice_score:.4f}"
        )

        scheduler.step()


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument("--data_path", type=str, help="path of the input data")
    parser.add_argument(
        "--epochs", "-e", type=int, default=500, help="number of epochs"
    )
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="batch size")
    parser.add_argument(
        "--learning-rate", "-lr", type=float, default=7e-4, help="learning rate"
    )
    parser.add_argument("--device", "-d", type=str, default="cuda:1", help="device")
    parser.add_argument("--seed", "-s", type=int, default=88, help="seed")
    parser.add_argument(
        "--model", "-m", type=str, default="resnet34_unet", help="model"
    )
    parser.add_argument("--wandb", "-w", type=bool, default=True, help="use wandb")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.wandb:
        wandb.init(
            project="NYCU_DL_Lab2",
            name=f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
        )
    train(args)
