from typing import Dict, List
from net.unet import UNet
from net.eunet import EUnet
from torch import nn
from spgutils.diceloss import BinaryDiceLoss
import focalloss
import torch
from os.path import join

name2model = {"unet": UNet, "eunet": EUnet}

name2act = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}


def create_model(cfg: Dict) -> nn.Module:
    cfg = cfg.copy()
    model_name = cfg.pop("name").lower()
    if "act" in cfg:
        act_name = cfg.pop("act")
        act = name2act[act_name]
        cfg["act"] = act

    model = name2model[model_name](**cfg)

    return model


name2criterion = {
    "bce": nn.BCEWithLogitsLoss,
    "bdice": BinaryDiceLoss,
    "focal": focalloss.FocalLossBinary,
}


def create_criterion(cfg: List[Dict]) -> nn.Module:
    cfg = cfg.copy()
    criterion_list = []
    weight_list = []
    for cfg_t in cfg:
        c_name = cfg_t.pop("name")
        weight = cfg_t.pop("weight")
        c = name2criterion[c_name]
        criterion_t = c(**cfg_t)
        criterion_list.append(criterion_t)
        weight_list.append(weight)

    class Criterion(nn.Module):
        def __init__(self):
            super().__init__()
            self.criterion_list = nn.ModuleList(criterion_list)
            self.weight_list = weight_list

        def forward(self, logits, targets):
            loss = 0
            for criterion_t, weight_t in zip(self.criterion_list, self.weight_list):
                loss += criterion_t(logits, targets) * weight_t
            return loss

    return Criterion()


from torch.optim import SGD, Adam, AdamW

name2opt = {"sgd": SGD, "adam": Adam, "adamw": AdamW}


def create_opt(cfg: Dict, params) -> torch.optim.Optimizer:
    cfg = cfg.copy()
    opt_name = cfg["name"]
    opt = name2opt[opt_name](lr=cfg["lr"], params=params)
    return opt


from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

name2scheduler = {"plat": ReduceLROnPlateau, "multistep": MultiStepLR}


def create_scheduler(cfg: Dict, opt: torch.optim.Optimizer):
    cfg = cfg.copy()
    name = cfg.pop("name")
    cfg["optimizer"] = opt
    scheduler = name2scheduler[name](**cfg)

    return scheduler


import datasetBUSI.weak_strong_busi
import datasetBUSI.mask_busi
from datasetBUSI.weak_strong_busi import WeakStrongBUSI
from datasetBUSI.mask_busi import MaskBUSI

name2dataset = {"weak_strong": WeakStrongBUSI, "mask": MaskBUSI}


def create_dataset(cfg: Dict):
    cfg = cfg.copy()
    root_dir = cfg["root_dir"]
    with open(join(root_dir, cfg["train_file"])) as file:
        train_filenames_l = file.read().split("\n")
    with open(join(root_dir, f"un{cfg['train_file']}")) as file:
        train_filenames_u = file.read().split("\n")
    with open(join(root_dir, "test.txt")) as file:
        test_filenames = file.read().split("\n")
    with open(join(root_dir, "val.txt")) as file:
        val_filenames = file.read().split("\n")

    name = cfg["name"]

    # trainset = name2dataset[name](root_dir, train_filenames, mode="train")
    trainset_l = MaskBUSI(root_dir, train_filenames_l, mode="train")
    trainset_u = WeakStrongBUSI(root_dir, train_filenames_u, mode="train")
    valset = MaskBUSI(root_dir, val_filenames, mode="test")
    testset = MaskBUSI(root_dir, test_filenames, mode="test")
    return (trainset_u, trainset_l), valset, testset
