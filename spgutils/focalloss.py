import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__()
        if alpha is None:
            alpha = [1, 1]
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        # 计算交叉熵损失
        ce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        # 计算焦距项
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        for i, t in enumerate(self.alpha):
            index = target == i
            focal_loss[index] = focal_loss[index] * t

        # 返回平均损失
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        if alpha is None:
            alpha = [1, 1, 1, 1]
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, target, reduction="none")
        # 计算焦距项
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        for i, t in enumerate(self.alpha):
            index = target == i
            focal_loss[index] *= t

        # 返回平均损失
        return focal_loss.mean()


if __name__ == "__main__":
    # loss = FocalLoss()
    # x = torch.Tensor([[10, 10]])
    # y = torch.LongTensor([0])
    # y2 = torch.LongTensor([1])
    # l1 = loss(x, y)
    # l2 = loss(x, y2)
    loss = BinaryFocalLoss()
    x = torch.Tensor([0])
    y = torch.Tensor([0])
    y2 = torch.Tensor([1])
    loss1 = loss(x, y)
    loss2 = loss(x, y2)
    pass
