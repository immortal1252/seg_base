import os

import torch
import yaml

import spgutils.log
from spgutils import utils
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Dict


class Pipeline:
    def __init__(self, config_path: str):
        self.path = config_path
        self.config: Dict = self.load(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = self.config
        if "model" in config:
            cfg = config["model"]
            self.model: nn.Module = utils.create(cfg["name"], **cfg.get("args", {})).to(
                self.device
            )
        if "trainset" in config:
            cfg = config["trainset"]
            self.trainset: Dataset = utils.create(cfg["name"], **cfg.get("args", {}))
        if "testset" in config:
            cfg = config["testset"]
            self.testset: Dataset = utils.create(cfg["name"], **cfg.get("args", {}))
        if "criterion" in config:
            cfg = config["criterion"]
            self.criterion: nn.Module = utils.create(
                cfg["name"], **cfg.get("args", {})
            ).to(self.device)
        if "optimizer" in config:
            cfg = config["optimizer"]
            self.optimizer: Optimizer = utils.create(
                cfg["name"], params=self.model.parameters(), **cfg.get("args", {})
            )
        if "scheduler" in config:
            cfg = config["scheduler"]
            self.scheduler: _LRScheduler = utils.create(
                cfg["name"], optimizer=self.optimizer, **cfg.get("args", {})
            )
        utils.seed_everything(42)
        self.logger = spgutils.log.logger
        self.logger.info(os.getpid())
        self.logger.info(self.path)
        self.logger.info(self.config)
        self.logger.info(utils.get_pararms_num(self.model))

    def train(self):
        raise NotImplementedError("not implement")

    def evaluate(self, test_loader: DataLoader):
        raise NotImplementedError("not implement")

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss

    @staticmethod
    def load(config_path: str):
        with open(config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
