import argparse

import torch
from torch.utils.data import DataLoader

from oxford_pet import load_dataset
from evaluate import evaluate

from models import ResNet34Unet, UNet

import matplotlib.pyplot as plt

def inference(model, device):
    if "resnet34" in args.model:
        model = ResNet34Unet()
    else:
        model = UNet()

    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    test_dataset = load_dataset(args.data_path, "test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    dice_score, _, _ = evaluate(model, test_loader, device=device)
    print(f"Dice score: {dice_score:.4f}")


def compare_predictions(model1, model2, device):
    if "resnet34" in args.model1:
        model1 = ResNet34Unet()
    else:
        model1 = UNet()

    if "resnet34" in args.model2:
        model2 = ResNet34Unet()
    else:
        model2 = UNet()

    model1.load_state_dict(torch.load(args.model1))
    model2.load_state_dict(torch.load(args.model2))
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    sample_count = 5

    test_dataset = load_dataset(args.data_path, "test")
    fig, axs = plt.subplots(4, sample_count, figsize=(sample_count * 10, 40))


    for i in range(sample_count):
        sample = test_dataset[i]
        image = sample["image"].to(device).unsqueeze(0)
        mask = sample["mask"]
        prediction1 = model1(image)
        prediction2 = model2(image)

        prediction1 = torch.where(prediction1 > 0.5, 1, 0)
        prediction2 = torch.where(prediction2 > 0.5, 1, 0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        image = image * std + mean

        axs[0, i].imshow(image.squeeze(0).moveaxis(0, 2).cpu().numpy())
        axs[1, i].imshow(mask.squeeze(0).cpu().numpy())
        axs[2, i].imshow(prediction1.squeeze(0).moveaxis(0, 2).cpu().numpy())
        axs[3, i].imshow(prediction2.squeeze(0).moveaxis(0, 2).cpu().numpy())
        axs[0, i].set_title("Image")
        axs[1, i].set_title("Mask")
        axs[2, i].set_title(f"Prediction from {model1.__class__.__name__}")
        axs[3, i].set_title(f"Prediction from {model2.__class__.__name__}")
    plt.tight_layout()
    plt.savefig(f"predictions_compare_{model1.__class__.__name__}_{model2.__class__.__name__}.png")
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model", default="MODEL.pth", help="path to the stored model weoght"
    )
    parser.add_argument("--data_path", type=str, help="path to the input data")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="batch size")

    parser.add_argument("--compare", "-c", action="store_true", help="compare the predictions")
    parser.add_argument("--model1", type=str, help="path to the stored model weoght")
    parser.add_argument("--model2", type=str, help="path to the stored model weoght")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    assert (args.compare and args.model1 and args.model2) or (not args.compare and args.model), "Either compare two models or predict a single model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.compare:
        compare_predictions(args.model1, args.model2, device)
    else:
        inference(args.model, device)
