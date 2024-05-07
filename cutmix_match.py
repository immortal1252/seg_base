from typing_extensions import override
from torch.utils.data import DataLoader
import abstract_unimatch
import torch

# TODO not implement yet
class CutmixMatch(abstract_unimatch.AbstractUnimatch):
    @override
    def train_epoch_semi(self, train_l_loader: DataLoader, train_u_loader: DataLoader):
        epoch_loss = 0
        loader = zip(train_l_loader, train_u_loader)
        self.model.train()
        for (x, y), (weak, mask, strong1, strong2, mix) in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            weak = weak.to(self.device)
            strong1 = strong1.to(self.device)
            strong2 = strong2.to(self.device)

            inputs = torch.cat([x, weak, strong1, strong2], dim=0)
            logits = self.model(inputs)
            y_logits, weak_logits, strong1_logits, strong2_logits = torch.split(
                logits, [x.shape[0], weak.shape[0], strong1.shape[0], strong2.shape[0]]
            )
            # supervised
            loss_x = self.criterion(y_logits, y).mean()

            # unsupervised
            y_logits_semi = weak_logits.detach()
            y_prob_semi = torch.sigmoid(y_logits_semi)
            mask_y = (y_prob_semi > 0.95) + (y_logits_semi < 0.05)
            y_pred_semi = y_prob_semi > 0.5
            loss_s1 = self.criterion(strong1_logits, y_pred_semi.float())
            loss_s2 = self.criterion(strong2_logits, y_pred_semi.float())
            loss_s1 = (loss_s1 * mask_y).mean()
            loss_s2 = (loss_s2 * mask_y).mean()

            loss = 2 * loss_x + loss_s1 + loss_s2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss
