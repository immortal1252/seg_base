import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset.DataSets import BreastDataset

import albumentations as A


dir_checkpoint = "lxb/checkpoints/trash"
train_path = "lxb/data/train"
val_path = "lxb/data/val"


def _get_train_transformers():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.6, rotate_limit=45, p=0.5
            ),
            A.ElasticTransform(
                p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
            ),
        ],
        p=0.2,
    )


def train_net(
    net=None,
    device=None,
    epochs: int = 5,
    batch_size: int = 5,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,  # 缩放 还没用
    amp: bool = False,
):
    # # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    n_train = len(os.listdir("lxb/data/train/train_3c_crop50"))
    n_val = len(os.listdir("lxb/data/val/val_3c_crop50"))

    # train_set = BreastDataset(train_path, phase='train', transforms=_get_train_transformers)
    train_set = BreastDataset(train_path, phase="train")
    val_set = BreastDataset(val_path, phase="val")

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision amp: {amp}
    """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9
    )
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=50
    )  # goal: maximize Dice score    一个指标停止改进时，降低学习率
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)  # 梯度缩放
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    best_train_dice = 0
    best_val_dice = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images = batch["image"]
                true_masks = batch["mask"]

                assert images.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) + dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, net.n_classes)
                        .permute(0, 3, 1, 2)
                        .float(),
                        multiclass=False,
                    )

                optimizer.zero_grad(set_to_none=True)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(
                    **{
                        "loss (batch)": loss.item(),
                        "lr:": optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )

                # Evaluation round
                # division_step = (n_train // (1 * batch_size))
                # if division_step > 0:
                #     if global_step % division_step == 0:
                # histograms = {}
                # for tag, value in net.named_parameters():
                #     tag = tag.replace('/', '.')
                #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                # val_score = evaluate(net, val_loader, device)
                # scheduler.step(val_score)
                #
                # logging.info('Validation Dice score: {}'.format(val_score))
                # experiment.log({
                #     'learning rate': optimizer.param_groups[0]['lr'],
                #     'validation Dice': val_score,
                #     # 'images': wandb.Image(images[0].cpu()),
                #     # 'masks': {
                #     #     'true': wandb.Image(true_masks[0].float().cpu()),
                #     #     'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                #     # },
                #     'step': global_step,
                #     'epoch': epoch,
                #     **histograms
                # })

        # train eval
        train_score = evaluate(net, train_loader, device)
        logging.info(f"Train Dice score: {train_score}, at epoch: {epoch}")
        if train_score > best_train_dice:
            best_train_dice = train_score
            logging.info(f"Best Train Dice score: {best_train_dice}, at epoch: {epoch}")

        # val eval
        val_score = evaluate(net, val_loader, device)
        scheduler.step(val_score)
        logging.info(f"Val Dice score: {val_score}, at epoch: {epoch}")
        if val_score > best_val_dice:
            best_val_dice = val_score
            logging.info(f"Best Val Dice score: {best_val_dice}, at epoch: {epoch}")

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(
                net.state_dict(),
                str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch)),
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--epochs", "-e", metavar="E", type=int, default=500, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-4,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file"
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=0.5,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=True, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--classes", "-c", type=int, default=2, help="Number of classes"
    )
    parser.add_argument("--seed", type=int, default=44, help="random seed")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_net()
    # mine
    # net = MyUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    # net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = UNet2(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = UNet3(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = SKUNet(n_channels=3, n_classes=args.classes, bilinear=False)  #纯skunet

    # 待测
    # 编码全是SK，测试哪里加SE比较好 然后主测ResSKUNet，LUNet
    # net = ResSKUNet(n_channels=3, n_classes=args.classes,bilinear=args.bilinear)

    # net = MyModel1(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = MyModel2(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = LUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = MyModel4(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = MyModel5(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    # 对比实验
    # net = AttU_Net(img_ch=3, output_ch=args.classes)
    # net = U_Net(img_ch=3, output_ch=args.classes)
