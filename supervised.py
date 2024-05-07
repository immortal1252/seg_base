import argparse

import pandas as pd
import torch

import spgutils.pipeline
import spgutils.log
import spgutils.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import spgutils.meter_queue


class Supervised(spgutils.pipeline.Pipeline):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    supervised = Supervised(args.path)
    supervised.train()
