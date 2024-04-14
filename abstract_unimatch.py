import argparse
from typing_extensions import override

import pandas as pd

import spgutils.pipeline
import spgutils.log
import spgutils.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import spgutils.meter_queue
import abc


# 半监督抽象类，提供了半监督训练的框架，定义了抽象方法train_epoch_semi，以及主函数
class AbstractUnimatch(spgutils.pipeline.Pipeline):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.trainset_u = spgutils.utils.create(
            self.config["trainset_u"]["name"],
            **self.config["trainset_u"]["args"],
        )

    @override
    def train(self):
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        # save_path = self.config["save_path"]
        train_l_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        train_u_loader = DataLoader(
            self.trainset_u, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        meter_queue = spgutils.meter_queue.MeterQueue(5)
        dice = -1
        for epoch in tqdm(range(epochs)):
            loss = self.train_epoch_semi(train_l_loader, train_u_loader)
            self.scheduler.step()
            self.logger.info(f"Epoch {epoch}  {loss}")
            if epoch == epochs - 1 or epoch % 10 == 1:
                dice = self.evaluate(test_loader)
                meter_queue.append(dice, epoch)

        self.logger.info(meter_queue.get_best_epoch())
        self.logger.info(meter_queue.get_best_val())

        df = pd.read_csv("./result.csv")
        df.loc[len(df)] = {"desc": self.path, "dice": dice}
        df.to_csv("./result.csv", index=False)

    @abc.abstractmethod
    def train_epoch_semi(self, train_l_loader: DataLoader, train_u_loader: DataLoader):
        raise NotImplementedError("not implement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--impl", type=str, required=True)
    args = parser.parse_args()
    match: AbstractUnimatch = spgutils.utils.create(args.impl, config_path=args.path)
    match.train()
