import math
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "focal_neg_loss_with_logits",
    "weighted_bce_with_logits",
]


def focal_neg_loss_with_logits(preds, gt, alpha=2, belta=4):
    """
    borrow from https://github.com/princeton-vl/CornerNet
    """

    preds = torch.sigmoid(preds)

    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

#     pos_inds = gt.gt(0)
#     neg_inds = gt.eq(0)

    neg_weights = torch.pow(1 - gt[neg_inds], belta)

    loss = 0
    pos_pred = preds[pos_inds]
    neg_pred = preds[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, alpha)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, alpha) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


# def weighted_bce_with_logits(out, gt, pos_w=1.0, neg_w=30.0):
#     pos_mask = torch.where(gt == 1, torch.ones_like(gt), torch.zeros_like(gt))
#     neg_mask = torch.ones_like(pos_mask) - pos_mask

#     losses = F.binary_cross_entropy_with_logits(out, gt, reduction='none')

#     loss_neg = (losses * neg_mask).sum() / (torch.sum(neg_mask))
#     loss_v = loss_neg * neg_w

#     pos_sum = torch.sum(pos_mask)
#     if pos_sum != 0:
#         loss_pos = (losses * pos_mask).sum() / pos_sum
#         loss_v += (loss_pos * pos_w)
#     return loss_v


def weighted_bce_with_logits(out, gt, pos_w=1.0, neg_w=30.0):
    pos_mask = torch.where(gt != 0.0, torch.ones_like(gt), torch.zeros_like(gt))
    #pos_mask = torch.where(gt == 1, torch.ones_like(gt), torch.zeros_like(gt))
    neg_mask = torch.ones_like(pos_mask) - pos_mask
    loss = F.binary_cross_entropy_with_logits(out, gt, reduction='none')
    loss_pos = (loss * pos_mask).sum() / ( torch.sum(pos_mask) + 1e-5)
    loss_neg = (loss * neg_mask).sum() / ( torch.sum(neg_mask) + 1e-5)
    loss = loss_pos * pos_w + loss_neg * neg_w
    return loss

