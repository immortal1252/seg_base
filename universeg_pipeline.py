import argparse

import torch
from torch.utils.data import DataLoader
import torch.utils.data.sampler
import random
import spgutils.pipeline
import spgutils.metric
from typing import Sized


class FixedLengthSampler(torch.utils.data.sampler.Sampler):
    """
    固定采样个数，随机打乱后循环采样
    """

    def __init__(self, dataset, num_samples):
        super().__init__(None)
        self.dataset = dataset
        self.num_samples = num_samples

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        cnt = 0
        while cnt < self.num_samples:
            yield indices[cnt % len(self.dataset)]
            cnt += 1

    def __len__(self):
        return self.num_samples


class UniversegPipeline(spgutils.pipeline.Pipeline):
    def prepare_data(self):
        batch_size = self.config["batch_size"]
        k = int(self.config["k"])
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        support_loader_list_train = [
            DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
            for _ in range(k)
        ]

        test_k = int(self.config.get("test_k", k))
        assert isinstance(self.testset, Sized)  # escape warning
        support_loader_list_test = [
            DataLoader(
                self.trainset,
                sampler=FixedLengthSampler(self.trainset, len(self.testset)),
                batch_size=batch_size,
            )
            for _ in range(test_k)
        ]

        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)

        if hasattr(self, "validset"):
            valid_loader = DataLoader(
                self.validset, batch_size=batch_size, shuffle=False
            )
            assert isinstance(self.validset, Sized)  # escape warning
            support_loader_list_valid = [
                DataLoader(
                    self.trainset,
                    sampler=FixedLengthSampler(self.trainset, len(self.validset)),
                    batch_size=batch_size,
                )
                for _ in range(test_k)
            ]
        else:
            valid_loader = None
            support_loader_list_valid = None
        return (
            (train_loader, support_loader_list_train),
            (
                test_loader,
                support_loader_list_test,
            ),
            (valid_loader, support_loader_list_valid),
        )

    def train_epoch(self, train_support_loader):
        train_loader, support_loader_list = train_support_loader
        support_loader = zip(*support_loader_list)
        self.model.train()
        epoch_loss = 0
        for (u, uy), (v_vy_tuples) in zip(train_loader, support_loader):
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
            torch.cuda.empty_cache()
            loss = self.criterion(logits, uy)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss

    def evaluate(self, test_support_loader):
        test_loader, support_loader_list = test_support_loader
        support_loader = zip(*support_loader_list)
        self.model.eval()
        metric_vector = dict()
        for batch_id, ((u, uy), (v_vy_tuples)) in enumerate(
            zip(test_loader, support_loader)
        ):
            u = u.to(self.device)
            uy = uy.to(self.device)

            vs = [v for v, vy in v_vy_tuples]
            vys = [vy for v, vy in v_vy_tuples]
            vs = torch.stack(vs, dim=1)
            vys = torch.stack(vys, dim=1)
            vs = vs.to(self.device)
            vys = vys.to(self.device)

            with torch.no_grad():
                y_pred = self.model(u, vs, vys)

            y_pred = torch.where(y_pred > 0, 1, 0)
            self.save_y_ypred(uy, y_pred, batch_id)
            with torch.no_grad():
                metric = spgutils.metric.compute_metric(y_pred, uy, self.all_metric)
            for k, v in metric.items():
                metric_vector.setdefault(k, []).append(v)

        dice = -1
        for k, v in metric_vector.items():
            values = torch.cat(v, 0)
            values_mean = values.mean(0).item()
            values_std = values.std(0).item()
            if k == "dice":
                dice = values_mean
            self.logger.info(f"{k}: {values_mean:.4}±{values_std:.4}")

        return dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    supervised = UniversegPipeline(args.path)
    supervised.train()
