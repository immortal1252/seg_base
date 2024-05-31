import os.path

import torch
import spgutils.metric

if __name__ == '__main__':
    x = torch.randint(0, 2, (3000, 1, 384, 384))
    y = torch.randint_like(x, 0, 2)
    metric = spgutils.metric.compute_metric(y, x, True)
    print(metric)
    pass
