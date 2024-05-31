from typing import Dict

import torch
from medpy.metric import hd95
import torch


def compute_metric(
    y_pred: torch.Tensor, y: torch.Tensor, all_metric=False
) -> Dict[str, torch.Tensor]:
    """
    计算一个小批次里每一张图片的指标
    Args:
        y_pred:bchw
        y: bchw
        all_metric:true计算一堆指标，否则只计算dice

    Returns:返回值尺寸(b)

    """
    ret = {}
    batch_size = y.shape[0]
    if all_metric:
        hd95_list = []
        try:
            for i in range(batch_size):
                hd = hd95(y_pred[i].cpu().numpy(), y[i].cpu().numpy())
                hd95_list.append(torch.Tensor([hd]))
            ret["hd95"] = torch.cat(hd95_list)
        except Exception as e:
            print(e)

    eps = 1e-10
    y = torch.flatten(y, 1)
    y_pred = torch.flatten(y_pred, 1)
    intersection = torch.sum(y * y_pred, 1)
    union_sum = torch.sum(y_pred + y, 1)
    dice = (2 * intersection + eps) / (union_sum + eps)
    ret["dice"] = dice
    if all_metric:
        union_or = torch.sum(torch.logical_or(y_pred, y), 1)
        tp = torch.sum((y == 1) == (y_pred == 1), 1)
        fp = torch.sum((y == 0) == (y_pred == 1), 1)
        tn = torch.sum((y == 0) == (y_pred == 0), 1)
        fn = torch.sum((y == 1) == (y_pred == 0), 1)
        jaccard = (intersection + eps) / (union_or + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        pre = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        spe = tn / (tn + fp + eps)
        ret["jaccard"] = jaccard
        ret["acc"] = acc
        ret["pre"] = pre
        ret["rec"] = rec
        ret["spe"] = spe

    return ret
