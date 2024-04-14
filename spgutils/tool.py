import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import time
from .move import move
import torch


def train_epoch(model: nn.Module, dataloder: DataLoader, opt: Optimizer, criterion):
    device = next(model.parameters()).device
    start = time.perf_counter()
    model.train()
    epoch_loss = 0
    for inputs, targets, *_ in dataloder:
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


def ema(teacher: nn.Module, student: nn.Module, beta: float = 0.999):
    # update weight
    with torch.no_grad():
        for param_student, param_teacher in zip(
            student.parameters(), teacher.parameters()
        ):
            param_teacher.data = param_teacher.data * beta + param_student.data * (
                1 - beta
            )
        # update bn
        for buffer_student, buffer_teacher in zip(student.buffers(), teacher.buffers()):
            buffer_teacher.data = buffer_teacher.data * beta + buffer_student.data * (
                1 - beta
            )
            buffer_teacher.data = buffer_student.data
