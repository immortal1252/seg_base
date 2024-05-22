import torch

from universeg_res.universeg import UniverSeg
from torchvision.models import ResNet

if __name__ == "__main__":
    import spgutils.utils

    arg_u = [4, 1, 384, 384]
    arg_v = [4, 2, 1, 384, 384]
    # b, l, d

    u = torch.randn(*arg_u)
    v = torch.randn(*arg_v)
    vy = torch.randn(*arg_v)
    model = UniverSeg([64, 128, 256, 256, 512], )
    y = model(u, v, vy)
    print(spgutils.utils.get_pararms_num(model))
    print(y.shape)
    pass
