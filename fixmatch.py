import argparse

import pandas as pd
import torch

import spgutils.pipeline
import spgutils.log
import spgutils.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import spgutils.meter_queue


class FixMatch(spgutils.pipeline.Pipeline):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.trainset_u = spgutils.utils.create(
            self.config["trainset_u"]["name"],
            **self.config["trainset_u"]["args"],
        )

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
        for epoch in tqdm(range(epochs)):
            loss = self.train_epoch_semi(train_l_loader, train_u_loader)
            self.scheduler.step()
            self.logger.info(f"Epoch {epoch}:{loss}")
            if epoch % 10 == 1:
                dice = self.evaluate(test_loader)
                meter_queue.append(dice, epoch)

        self.logger.info(meter_queue.get_best_epoch())
        self.logger.info(meter_queue.get_best_val())

        df = pd.result.csv("./result.csv")
        df.loc[len(df)] = {"desc": self.config["desc"], "dice": dice}
        df.to_csv("./result.csv", index=False)

    def train_epoch_semi(self, train_l_loader: DataLoader, train_u_loader: DataLoader):
        epoch_loss = 0
        loader = zip(train_l_loader, train_u_loader)
        self.model.train()
        for (x, y), (weak, mask, strong1) in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            weak = weak.to(self.device)
            strong1 = strong1.to(self.device)

            inputs = torch.cat([x, weak, strong1], dim=0)
            logits = self.model(inputs)
            y_logits, weak_logits, strong1_logits = torch.chunk(logits, 3, dim=0)
            # supervised
            loss_x = self.criterion(y_logits, y).mean()

            # unsupervised
            y_logits_semi = weak_logits.detach()
            y_prob_semi = torch.sigmoid(y_logits_semi)
            mask_y = (y_prob_semi > 0.95) + (y_logits_semi < 0.05)
            y_pred_semi = y_prob_semi > 0.5
            loss_s1 = self.criterion(strong1_logits, y_pred_semi.float())
            loss_s1 = (loss_s1 * mask_y).mean()

            loss = 2 * loss_x + loss_s1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss

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
    supervised = FixMatch(args.path)
    supervised.train()
