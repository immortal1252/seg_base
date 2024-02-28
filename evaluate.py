import torch
import factory
from torch.utils.data import DataLoader
from torch import nn
import yaml
import datasetBUSI.base_busi as base_busi
from net.eunet import EUnet
from net.unet import UNet
from spgutils.segmetric import compute_dice, Meter


def debug(tensor, name="new.png"):
    import cv2
    from PIL import Image
    from torchvision.transforms import ToPILImage

    pil = ToPILImage()(tensor)
    pil.save(name)


def eval(model, testloader, criterion=None):
    device = next(model.parameters())
    model.eval()
    meter = Meter(pre=2)
    loss = 0
    cnt = 0
    for x, y, *_ in testloader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = model(x)
        if criterion is not None:
            loss_t = criterion(logits, y).item()
            loss += loss_t
        dice = compute_dice(logits, y)
        if dice == 0:
            cnt += 1

        meter_t = {"dice": dice}
        meter += meter_t
    print(cnt)
    return meter.mean(), loss


if __name__ == "__main__":
    from net.unet import UNet

    with open("configs-bak/semi_base.yaml") as file:
        cfg = yaml.load(file, yaml.FullLoader)

    model = factory.create_model(cfg["model"])
    from os.path import join

    # model.load_state_dict(torch.load("20base/final.pt"))
    model.load_state_dict(torch.load("semi/final.pt"))
    model = model.to("cuda")

    trainset, testset, valset = factory.create_dataset(cfg["dataset"])
    trainloader = DataLoader(trainset, batch_size=8)
    testloader = DataLoader(testset, batch_size=8)
    valloader = DataLoader(valset, batch_size=8)
    dicetrain, _ = eval(model, trainloader)
    dicetest, _ = eval(model, testloader)
    diceval, _ = eval(model, valloader)
    print(dicetrain)
    print(dicetest)
    print(diceval)
