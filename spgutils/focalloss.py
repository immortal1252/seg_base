from torch import nn
import torch
import torch.nn.functional as F


class FocalLossBinary(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLossBinary, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # 计算二分类交叉熵损失
        ce_loss = F.binary_cross_entropy(logits, target, reduction="none")

        # 计算焦距项
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # 返回损失
        return focal_loss.mean()  # 可以使用 mean 或 sum 汇总损失
