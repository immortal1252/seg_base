import torch
import torch.nn as nn


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = list(input.shape)
    shape[1] = num_classes

    one_hot = torch.zeros(shape).to(input.device)

    one_hot.scatter_(1, input, 1)

    return one_hot


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        """
        output need sigmoid withou sigmoid
        output: [N, 1, *] without activation
        target:  [N, 1, *] without one hot
        """
        super().__init__()

    def forward(self, output, target):
        assert output.shape == target.shape, f"output.shape({output.shape}) != target.shape({target.shape})"

        output = torch.sigmoid(output)
        output = output.contiguous().view(output.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        smooth = 1
        num = 2 * torch.sum(output * target, 1) + smooth
        den = torch.sum(output + target, 1) + smooth

        loss = 1 - num / den

        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        """
        output need softmax without one-hot and softmax
        output: [N, C, *] without activation
        target:  [N, 1, *] without one hot
        """
        super(DiceLoss, self).__init__()
        self.bdice = BinaryDiceLoss()
        self.num_classes = num_classes

    def forward(self, output: torch.Tensor, target: torch.LongTensor):
        target = make_one_hot(target, self.num_classes)

        assert output.shape == target.shape, f"output.shape({output.shape}) != target.shape({target.shape})"

        output = torch.softmax(output, 1)
        total_loss = 0

        for i in range(target.shape[1]):
            dice_loss = self.bdice(output[:, i], target[:, i])
            total_loss += dice_loss

        return total_loss / target.shape[1]
