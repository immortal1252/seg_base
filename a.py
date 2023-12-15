import net.unet.unet_model
import torch
from spgutils import get_pararms_num
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import cv2
import factory
import yaml

if __name__ == "__main__":
    result = {
        "model": "model",
        "final_dice": 0,
        "best_dice": 0,
        "opt": 0,
        "base_lr": 0,
        "scheduler": 0,
        "config": 0,
        "other": 0,
    }

    import pandas as pd

    pd.DataFrame([result]).to_csv("1.csv")
    # df = pd.read_csv("result.csv")
    # df.loc[len(df)] = result
    # df.to_csv("result.csv", index=False)
