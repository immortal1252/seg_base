import argparse

import torch
import torch.utils.data.sampler

import universeg_pipeline


class UniversegClaPipeline(universeg_pipeline.UniversegPipeline):

    def evaluate(self, test_support_loader):
        test_loader, support_loader_list = test_support_loader
        support_loader = zip(*support_loader_list)
        self.model.eval()
        tp = [0, 0]
        pixel_cnt = [0, 0]
        intersection = [0, 0]
        union = [0, 0]
        for batch_id, ((u, uy, cla), (v_vy_tuples)) in enumerate(
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

            for i in range(2):
                idx = torch.nonzero(cla == i).flatten()
                intersection[i] += torch.sum(uy[idx] * y_pred[idx]).item()
                union[i] += torch.sum(y_pred[idx] + uy[idx]).item()
                tp[i] += torch.sum(y_pred[idx] == uy[idx]).item()
                pixel_cnt[i] += uy[idx].numel()

        for i in range(2):
            dice_avg_t = (2 * intersection[i] + 1e-9) / (union[i] + 1e-9)
            acc_avg_t = tp[i] / pixel_cnt[i]
            self.logger.info(f"{i}")
            self.logger.info(f"Dice score: {dice_avg_t:.4}")
            self.logger.info(f"Accuracy: {acc_avg_t:.4}")

        dice_avg = (2 * sum(intersection) + 1e-9) / (sum(union) + 1e-9)
        return dice_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    supervised = UniversegClaPipeline(args.path)
    supervised.train()
