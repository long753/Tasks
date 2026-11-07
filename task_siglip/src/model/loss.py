import torch
import torch.nn.functional as F
from torch._tensor import Tensor


def siglip_cross_entropy_loss(logits, positive_mask):
    """
    Computes the cross-entropy loss for SigLIP model.

    Args:
        logits (torch.Tensor): Logits of the similarity matrix.
        positive_mask (torch.Tensor): The correct text-image pair mask.

    Returns:
        loss: The siglip cross-entropy loss.
    """

    # 合理性检验
    if positive_mask is None:
        raise ValueError("positive_mask must be provided")
    if not isinstance(positive_mask, torch.Tensor):
        raise TypeError("positive_mask must be a tensor")
    if positive_mask.shape != logits.shape:
        raise ValueError("positive_mask must have the same shape as logits")
    if positive_mask.dtype != torch.bool and not torch.all(input=(positive_mask == 0) | (positive_mask == 1)).item():
        raise ValueError("positive_mask values must be boolean or 0/1")

    # 计算 loss
    labels = positive_mask.to(device=logits.device, dtype=logits.dtype) * 2 - 1
    loss_matrix: Tensor = -F.logsigmoid(input=labels * logits)
    loss: Tensor = loss_matrix.sum() / logits.size(0)

    return loss
