import argparse

import torch
from torch.utils.data import DataLoader

import spgutils.pipeline


class UniversegPipeline(spgutils.pipeline.Pipeline):
    def prepare_data(self):
        batch_size = self.config["batch_size"]
        k = int(self.config["k"])
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        support_loader = zip(*[DataLoader(self.trainset, batch_size=batch_size, shuffle=True) for _ in range(k)])
        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        train_support_loader = zip(train_loader, support_loader)
        return train_support_loader, test_loader

    def train_epoch(self, train_support_loader):
        self.model.train()
        epoch_loss = 0
        for (u, uy), (v_vy_tuples) in train_support_loader:
            # kv:(v1,vy2),(v2,vy2)...(vk,vyk)
            u = u.to(self.device)
            uy = uy.to(self.device)

            vs = [v for v, vy in v_vy_tuples]
            vys = [vy for v, vy in v_vy_tuples]
            vs = torch.stack(vs, dim=1)
            vys = torch.stack(vys, dim=1)
            vs = vs.to(self.device)
            vys = vys.to(self.device)

            logits = self.model(u, vs, vys)
            loss = self.criterion(logits, uy)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    supervised = UniversegPipeline(args.path)
    supervised.train()
