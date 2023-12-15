import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import time
from .move import move


def train_epoch(model: nn.Module, dataloder: DataLoader, opt: Optimizer, criterion):
    device = next(model.parameters()).device
    start = time.perf_counter()
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
    end = time.perf_counter()
    return epoch_loss, end - start
