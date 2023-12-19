from os.path import join
from typing import Dict
import torch
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import factory

# from net.unet import UNet
from torch.utils.data import DataLoader
import spgutils.tool
from spgutils.meter_queue import MeterQueue
from spgutils.log import logger

from spgutils import device
import evaluate
import datasetBUSI.base_busi as base_busi
import yaml
import argparse
import os
import semi.tool

if __name__ == "__main__":
    spgutils.seed_everything(42)

    logger.info(f"pid={os.getpid()}")
    # 加载配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--other", type=str, required=False, help="其他描述信息")
    args = parser.parse_args()
    with open(args.config) as file:
        cfg: Dict = yaml.load(file, Loader=yaml.FullLoader)
    logger.info(f"config_path:{args.config}")
    logger.info(cfg)
    ori_cfg = cfg.copy()
    # 加载数据集
    (trainset_u, trainset_l), testset, valset = factory.create_dataset(cfg["dataset"])
    trainloader_u = DataLoader(trainset_u, cfg["batch_size"], True)
    trainloader_l = DataLoader(trainset_l, cfg["batch_size"], True)
    testloader = DataLoader(testset, 8, False)
    valloader = DataLoader(valset, 8, False)

    # 加载模型
    model = factory.create_model(cfg["model"]).to(device)
    opt = factory.create_opt(cfg["opt"], model.parameters())
    criterion = factory.create_criterion(cfg["criterion"]).to(device)
    scheduler = factory.create_scheduler(cfg["scheduler"], opt)
    os.makedirs(cfg["save_path"], exist_ok=True)

    if "semi" in cfg.keys():
        model.load_state_dict(torch.load(cfg["semi"]["pretrained"]))
        model_ema = factory.create_model(cfg["model"]).to(device)
        model_ema.load_state_dict(model.state_dict())

    # 训练
    meter_queue = MeterQueue(5)
    max_dice = 0
    global_step = 0
    for epoch in tqdm(range(cfg["epochs"])):
        if epoch >= cfg["valid_start_epoch"]:
            meter, eval_loss = evaluate.eval(model, valloader, criterion)
            dice = meter["dice"]
            if dice > max_dice:
                max_dice = dice
                torch.save(model.state_dict(), join(cfg["save_path"], "best.pt"))

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(dice)
            meter_queue.append(dice, epoch)
            logger.info(f"dice: {dice}")

        if "semi" in cfg.keys():
            loss, ratio, global_step = semi.tool.train_epoch(
                model, model_ema, trainloader_u, trainset_l, opt, global_step
            )
            logger.info(f"{epoch}: u {loss}")
        loss, _ = spgutils.train_epoch(model, trainloader_l, opt, criterion)
        logger.info(f"{epoch}: l {loss}")
        if not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

    logger.info(f"bestepoch:{meter_queue.get_best_epoch}")
    logger.info(f"bestval:{meter_queue.get_best_val}")
    torch.save(model.state_dict(), join(cfg["save_path"], "final.pt"))

    import evaluate

    final_meter, _ = evaluate.eval(model, testloader)
    logger.info(f"final_meter:{final_meter}")

    model.load_state_dict(torch.load(join(cfg["save_path"], "best.pt")))
    best_meter, _ = evaluate.eval(model, testloader)
    logger.info(f"best_meter:{best_meter}")

    result = {
        "model": model.__class__.__name__,
        "final_dice": final_meter["dice"],
        "best_dice": best_meter["dice"],
        "opt": opt.__class__.__name__,
        "base_lr": ori_cfg["opt"]["lr"],
        "scheduler": scheduler.__class__.__name__,
        "config": args.config,
        "other": args.other,
    }

    if os.path.exists("result.csv"):
        df = pd.read_csv("result.csv")
        df.loc[len(df)] = result  # type: ignore
    else:
        df = pd.DataFrame([result])

    df.to_csv("result.csv", index=False)
