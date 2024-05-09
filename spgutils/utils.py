import importlib

import torch
from torch import nn
import numpy as np
import random
import torch.cuda
import torch.backends.cudnn
import os

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_pararms_num(model: nn.Module):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params / (1024 * 1024)


def move(m, device):
    if isinstance(m, torch.Tensor):
        m = m.to(device)
    elif isinstance(m, tuple):
        m = tuple(move(m_t, device) for m_t in m)
    elif isinstance(m, list):
        m = [move(m_t, device) for m_t in m]
    elif isinstance(m, dict):
        m = {k: move(v, device) for k, v in m.items()}
    else:
        raise Exception(f"m should be either tensor,tuple,list or dict,got{type(m)}")

    return m


def train_epoch(
    model: nn.Module, dataloder: DataLoader, opt: Optimizer, criterion: nn.Module
):
    device = next(model.parameters()).device
    model.train()
    epoch_loss = 0
    for inputs, targets in dataloder:
        inputs = move(inputs, device)
        targets = move(targets, device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    return epoch_loss


def init_params(model: nn.Module, nonlinearity="relu"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def create(full_class_name: str, **kwargs):
    split_index = full_class_name.rfind(".")
    module_name = full_class_name[:split_index]
    class_name = full_class_name[split_index + 1 :]
    module = importlib.import_module(module_name)
    obj = getattr(module, class_name)(**kwargs)
    return obj
