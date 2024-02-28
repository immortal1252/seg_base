import argparse

import pandas as pd
import torch

import spgutils.pipeline
import spgutils.log
import spgutils.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import spgutils.meter_queue


class Supervised(spgutils.pipeline.Pipeline):
    def __init__(self, config_path):
        super().__init__(config_path)
        # self.dataset_unlabeled = spgutils.utils.create(self.config["dataset_unlabeled"]["name"],
        #                                                **self.config["dataset_unlabeled"]["args"])

    def train(self):
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        # save_path = self.config["save_path"]
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        meter_queue = spgutils.meter_queue.MeterQueue(5)
        for epoch in tqdm(range(epochs)):
            loss = self.train_epoch(train_loader)
            self.scheduler.step()
            self.logger.info(f"Epoch {epoch}:{loss}")
            if epoch % 10 == 1:
                dice = self.evaluate(test_loader)
                meter_queue.append(dice, epoch)

        self.logger.info(meter_queue.get_best_epoch())
        self.logger.info(meter_queue.get_best_val())

        df = pd.result.csv("./result.csv")
        df.loc[len(df)] = {"desc": self.path[: self.path.rfind(".")], "dice": dice}
        df.to_csv("./result.csv", index=False)

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()
        dice_total = 0
        acc_total = 0
        cnt = 0
        for x, y in test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                y_pred = self.model(x)

            y_pred = torch.where(y_pred > 0, 1, 0)
            dice = (2 * torch.sum(y * y_pred) + 1e-9) / (torch.sum(y_pred + y) + 1e-9)
            acc = torch.sum(y_pred == y)
            cnt += x.shape[0]
            dice_total += dice.item()
            acc_total += acc.item()

        dice_avg = dice_total / cnt
        acc_avg = acc_total / cnt
        self.logger.info(f"Dice score: {dice_avg:.4}")
        self.logger.info(f"Accuracy: {acc_avg:.4}")

        return dice_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    supervised = Supervised(args.path)
    supervised.train()
