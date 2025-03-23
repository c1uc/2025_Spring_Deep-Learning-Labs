import argparse

import torch
from torch.utils.data import DataLoader

from oxford_pet import load_dataset
from evaluate import evaluate

from models import ResNet34Unet, UNet


def inference(model, device):
    if "resnet34" in args.model:
        model = ResNet34Unet(in_channels=3)
    else:
        model = UNet(in_channels=3)

    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    test_dataset = load_dataset(args.data_path, "test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    dice_score, _, _ = evaluate(model, test_loader, device=device)
    print(f"Dice score: {dice_score:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model", default="MODEL.pth", help="path to the stored model weoght"
    )
    parser.add_argument("--data_path", type=str, help="path to the input data")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="batch size")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference(args.model, device)
