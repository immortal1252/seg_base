import os

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from datasetBUSI.base_busi import save_img_from_tensor
import spgutils.meter_queue
import spgutils.log
from spgutils import utils
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Dict
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
        self.logger.info(type(self))
        self.logger.info(self.path)
        self.logger.info(self.config)
        self.logger.info(utils.get_pararms_num(self.model))

    def prepare_data(self):
        batch_size = self.config["batch_size"]
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def train(self):
        train_loader, test_loader = self.prepare_data()
        epochs = self.config["epochs"]
        dice = -1

        meter_queue = spgutils.meter_queue.MeterQueue(5)
        for epoch in tqdm(range(epochs)):
            loss = self.train_epoch(train_loader)
            self.scheduler.step()
            self.logger.info(f"Epoch {epoch}  {loss}")
            if epoch == epochs - 1 or epoch % 10 == 1:
                dice = self.evaluate(train_loader)
                dice = self.evaluate(test_loader)
                meter_queue.append(dice, epoch)

        self.post(meter_queue, dice)

    def post(self, meter_queue, dice):
        self.logger.info(meter_queue.get_best_epoch())
        self.logger.info(meter_queue.get_best_val())

        df = pd.read_csv("./result.csv")
        df.loc[len(df)] = {"desc": self.path[: self.path.rfind(".")], "dice": dice}
        df.to_csv("./result.csv", index=False)

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()
        tp = 0
        pixel_cnt = 0
        intersection = 0
        union = 0
        for batch_id, (x, y) in enumerate(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                y_pred = self.model(x)

            y_pred = torch.where(y_pred > 0, 1, 0)
            self.save_y_ypred(y, y_pred, batch_id)

            intersection += torch.sum(y * y_pred).item()
            union += torch.sum(y_pred + y).item()
            tp += torch.sum(y_pred == y).item()
            pixel_cnt += y.numel()

        dice_avg = (2 * intersection + 1e-9) / (union + 1e-9)
        acc_avg = tp / pixel_cnt
        self.logger.info(f"Dice score: {dice_avg:.4}")
        self.logger.info(f"Accuracy: {acc_avg:.4}")

        return dice_avg

    def save_y_ypred(self, y: torch.Tensor, ypred: torch.Tensor, id: int):
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
        with open(config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)


if __name__ == "__main__":
    a = Pipeline("a")
