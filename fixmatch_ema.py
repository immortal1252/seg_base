from copy import deepcopy
from typing_extensions import override

import torch
from spgutils.tool import ema
from torch.utils.data import DataLoader
import abstract_unimatch


class FixMatchEMA(abstract_unimatch.AbstractUnimatch):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.teacher = deepcopy(self.model)

    @override
    def train_epoch_semi(self, train_l_loader: DataLoader, train_u_loader: DataLoader):
        epoch_loss = 0
        loader = zip(train_l_loader, train_u_loader)
        self.model.train()
        self.teacher.eval()
        for (x, y), (weak, mask, strong1) in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            weak = weak.to(self.device)
            strong1 = strong1.to(self.device)

            with torch.no_grad():
                y_fake = self.teacher(weak.detach())

            inputs = torch.cat([x, strong1], dim=0)
            logits = self.model(inputs)
            y_logits, strong1_logits = torch.split(
                logits, [x.shape[0], strong1.shape[0]], dim=0
            )
            # supervised
            loss_x = self.criterion(y_logits, y).mean()

            # unsupervised
            y_prob_semi = torch.sigmoid(y_fake)
            mask_y = (y_prob_semi > 0.95) + (y_prob_semi < 0.05)
            y_pred_semi = y_prob_semi > 0.5
            loss_s1 = self.criterion(strong1_logits, y_pred_semi.float())
            loss_s1 = (loss_s1 * mask_y).mean()

            loss = 2 * loss_x + loss_s1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            ema(self.teacher, self.model)

        return epoch_loss
