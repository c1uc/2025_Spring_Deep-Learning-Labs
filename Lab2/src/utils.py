import torch


def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    assert pred_mask.shape == gt_mask.shape

    if pred_mask.ndim == 3:
        pred_mask = pred_mask.unsqueeze(1)

    pred_mask = torch.where(pred_mask > 0.5, True, False)
    gt_mask = torch.where(gt_mask > 0.5, True, False)

    common = torch.sum(pred_mask & gt_mask, dim=(1, 2, 3))
    union = torch.sum(pred_mask, dim=(1, 2, 3)) + torch.sum(gt_mask, dim=(1, 2, 3))

    union = torch.where(union == 0, 1, union)

    return (2 * common / union).mean()


def dice_loss(pred_mask, gt_mask):
    return 1 - dice_score(pred_mask, gt_mask)
