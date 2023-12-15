import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .tool import train_epoch
from .params import get_pararms_num
from .seed import seed_everything
