import os
import logging
import pandas as pd
import torch
import yaml
from tqdm import tqdm
import spgutils.metric
import spgutils.meter_queue
import spgutils.log
from spgutils import utils
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Dict, Union
from PIL import Image


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
        if "validset" in config:
            cfg = config["validset"]
            self.validset: Dataset = utils.create(cfg["name"], **cfg.get("args", {}))
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
            self.scheduler: Union[LRScheduler, ReduceLROnPlateau] = utils.create(
                cfg["name"], optimizer=self.optimizer, **cfg.get("args", {})
            )
            if isinstance(self.scheduler, ReduceLROnPlateau) and not hasattr(
                self, "validset"
            ):
                raise Exception("use ReduceLROnPlateau must provide validset")

        self.all_metric = config.get("all_metric", False)
        utils.seed_everything(42)
        self.log_dir = os.path.join(os.path.dirname(self.path), "log")
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.logger = spgutils.log.new_logger(self.log_dir)
        self.logger.info(os.getpid())
        self.logger.info(type(self))
        self.logger.info(self.path)
        self.logger.info(self.config)
        self.logger.info(utils.get_pararms_num(self.model))

        if self.config.get("init", False):
            act = self.config["model"].get("args", {}).get("act", "ReLU")
            if act == "LeakyReLU":
                act = "leaky_relu"
            elif act == "ReLU":
                act = "relu"
            else:
                raise Exception(f"no support init {act}")
            utils.init_params(self.model, act)
            self.logger.info(f"init with {act}")

    def prepare_data(self):
        batch_size = self.config["batch_size"]
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        if hasattr(self, "validset"):
            valid_loader = DataLoader(
                self.validset, batch_size=batch_size, shuffle=False
            )
        else:
            valid_loader = None
        return train_loader, test_loader, valid_loader

    def train(self):
        train_loader, test_loader, valid_loader = self.prepare_data()
        epochs = self.config["epochs"]
        valid_max_dice = -1  # 如果有验证集，使用验证集上的效果最好的那次进行测试，否则使用最好一次
        valid_best_checkpoint = None

        test_dice_list = []
        valid_dice_list = []
        self.evaluate(test_loader)
        for epoch in tqdm(range(epochs)):
            loss = self.train_epoch(train_loader)
            self.logger.info(f"Epoch {epoch}  {loss}")
            if hasattr(self, "scheduler"):
                if (
                    isinstance(self.scheduler, ReduceLROnPlateau)
                    and epoch >= epochs * 0.3
                ):
                    assert valid_loader is not None  # escape warning
                    self.logger.info("valid")
                    dice = self.evaluate(valid_loader)
                    valid_dice_list.append(dice)
                    if dice > valid_max_dice:
                        valid_max_dice = dice
                        valid_best_checkpoint = self.model.state_dict()

                    old_lr = self.optimizer.param_groups[0]["lr"]
                    self.scheduler.step(dice)
                    new_lr = self.optimizer.param_groups[0]["lr"]
                    if old_lr != new_lr:
                        self.logger.info(f"{old_lr}->{new_lr}")

                elif isinstance(self.scheduler, LRScheduler):
                    self.scheduler.step()

            if self.config.get("debug", False) and epoch >= epochs * 0.3:
                self.logger.debug("test")
                dice = self.evaluate(test_loader)
                test_dice_list.append(dice)

            self.logger.info("*" * 80)

        self.logger.debug(valid_dice_list)
        self.logger.debug(test_dice_list)

        if valid_best_checkpoint is not None:
            self.model.load_state_dict(valid_best_checkpoint)
        self.logger.info("final test")
        final_dice = self.evaluate(test_loader)
        self.post(final_dice, valid_best_checkpoint)

    def post(self, dice, checkpoint=None):
        df = pd.read_csv("./result.csv")
        log_name = ""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_name = handler.baseFilename

        df.loc[len(df)] = {
            "desc": self.path[: self.path.rfind(".")],
            "dice": dice,
            "log": log_name,
        }
        df.to_csv("./result.csv", index=False)
        if checkpoint is not None:
            torch.save(
                checkpoint,
                os.path.join(self.log_dir, str(self.model.__class__) + ".pt"),
            )

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()
        metric_vector = dict()
        for batch_id, (x, y) in enumerate(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                y_pred = self.model(x)

            y_pred = torch.where(y_pred > 0, 1, 0)
            self.save_y_ypred(y, y_pred, batch_id)
            with torch.no_grad():
                metric = spgutils.metric.compute_metric(y_pred, y, self.all_metric)

            for k, v in metric.items():
                metric_vector.setdefault(k, []).append(v)

        dice = -1
        for k, v in metric_vector.items():
            values = torch.cat(v, 0)
            values_mean = values.mean(0)
            values_std = values.std(0)
            if k == "dice":
                dice = values_mean
            self.logger.info(f"{k}: {values_mean:.4}±{values_std:.4}")

        return dice

    def save_y_ypred(self, y: torch.Tensor, ypred: torch.Tensor, id: int):
        if not os.path.exists("temp"):
            os.mkdir("temp")
        y = y.cpu().float()
        ypred = ypred.cpu().float()
        for i in range(y.shape[0]):
            topil = transforms.ToPILImage()
            png_y: Image.Image = topil(y[i])
            png_ypred = topil(ypred[i])
            png_y.save(f"temp/{id}_{i}_y.png")
            png_ypred.save(f"temp/{id}_{i}_ypred.png")

    def train_epoch(self, train_loader):
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
        with open(config_path, "r", encoding="utf8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
