import torch
import numpy as np

from utils import dice_score, dice_loss


def evaluate(net, data, device):
    # implement the evaluation function here
    net.eval()
    dice_scores = []
    bce_losses = []
    dice_losses = []
    bce_loss = torch.nn.BCELoss()
    with torch.no_grad():
        for batch in data:
            images = batch["image"].to(device).float()
            masks = batch["mask"].to(device)
            pred_masks = net(images)
            dice_scores.append(dice_score(pred_masks, masks).item())
            bce_losses.append(bce_loss(pred_masks, masks).item())
            dice_losses.append(dice_loss(pred_masks, masks).item())
    return np.mean(dice_scores), np.mean(bce_losses), np.mean(dice_losses)
