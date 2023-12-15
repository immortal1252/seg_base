from operator import mod
import torch
import torch.nn as nn


def get_pararms_num(model:nn.Module):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params/(1024*1024)