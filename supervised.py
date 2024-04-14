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


    def train(self):
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        # save_path = self.config["save_path"]
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        meter_queue = spgutils.meter_queue.MeterQueue(5)
        dice = -1
        for epoch in tqdm(range(epochs)):
            loss = self.train_epoch(train_loader)
            self.scheduler.step()
            self.logger.info(f"Epoch {epoch}  {loss}")
            if epoch == epochs - 1 or epoch % 10 == 1:
                dice = self.evaluate(test_loader)
                meter_queue.append(dice, epoch)

        self.logger.info(meter_queue.get_best_epoch())
        self.logger.info(meter_queue.get_best_val())

        df = pd.read_csv("./result.csv")
        df.loc[len(df)] = {"desc": self.path[: self.path.rfind(".")], "dice": dice}
        df.to_csv("./result.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    supervised = Supervised(args.path)
    supervised.train()
