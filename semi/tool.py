# 最朴素的实现
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import time
import aug
import torch.nn.functional as F
from .spgutils.move import move
import time


def get_dataloader(dataset: Dataset):
    pass


def ema(teacher: nn.Module, student: nn.Module, beta: float):
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


def compute_unsupervised_loss_by_threshold(logits_student, logits_teacher, thresh=0.95):
    # 返回损失,以及教师网络中合格的像素比例
    batch_size, num_class, h, w = logits_student.shape
    # target大于0.95表示高置信度目标
    mask = torch.abs(logits_teacher) > thresh
    target = logits_teacher
    target[logits_teacher > thresh] = 1
    target[logits_teacher < -thresh] = 0

    loss = F.binary_cross_entropy_with_logits(logits_student, target, weight=mask)
    # loss2 = F.binary_cross_entropy_with_logits(logits_student, target, weight=target)
    return loss, target.mean()


def debug(tensor, name="new.png"):
    import cv2
    from PIL import Image
    from torchvision.transforms import ToPILImage

    pil = ToPILImage()(tensor.cpu().float())
    pil.save(name)


def train_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataloder: DataLoader,
    opt: Optimizer,
    global_step,
    base_beta=0.9,
):
    device = next(student_model.parameters()).device
    student_model.train()
    teacher_model.eval()
    epoch_loss = 0
    epoch_ratio = 0
    # elapsed = 0
    for weak, mask, strong in dataloder:
        weak = move(weak, device)
        strong = move(strong, device)
        mask = move(mask, device)
        logits_student = student_model(strong)
        with torch.no_grad():
            logits_teacher = teacher_model(weak).detach()

        loss, ratio = compute_unsupervised_loss_by_threshold(
            logits_student, logits_teacher, 0.95
        )
        # loss = F.binary_cross_entropy_with_logits(logits_student, mask)
        ratio = 0
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
        epoch_ratio += ratio

        beta = min(base_beta, 1 - 1 / (global_step + 1))
        # start = time.perf_counter()
        ema(teacher_model, student_model, beta)
        # end = time.perf_counter()
        # elapsed += end - start

        global_step += 1

    # print(elapsed)
    return epoch_loss, epoch_ratio / len(dataloder), global_step
