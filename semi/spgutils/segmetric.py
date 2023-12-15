import math
from typing import Dict
import torch


class Meter:
    def __init__(self, keys=["dice"], pre=4):
        self._data = {key: 0 for key in keys}
        # 添加的时候自动累加，便于求平均值
        self._len = 0
        # str保留几位小数点
        self._pre = pre

    def __getitem__(self, key):
        assert key in self._data.keys(), "bad key"
        return self._data[key]

    def __iadd__(self, other: Dict):
        self._len += 1
        for key in other.keys():
            if key in self._data:
                self._data[key] += other[key]

        return self

    def mean(self):
        # 返回平均值，为了防止多次调用出问题，返回副本
        ret = Meter()
        ret._len = 1
        ret._data = self._data.copy()
        for key in ret._data.keys():
            ret._data[key] /= self._len  # type: ignore

        return ret

    def __str__(self):
        map_float2 = self._data.copy()
        for key in map_float2.keys():
            value = map_float2[key]
            if isinstance(value, float) and not math.isnan(value):
                # 保留两位小数
                newvalue = round(value, self._pre)
                map_float2[key] = newvalue

        return str(map_float2)


# targets只包含0或1,且0是背景,1是目标
def compute_dice(logits: torch.Tensor, targets: torch.Tensor):
    logits = (logits > 0).float()
    logits = torch.flatten(logits, 1)
    targets = torch.flatten(targets, 1)
    smooth = 1e-5
    num = (logits * targets).sum().item()
    den = (logits + targets).sum().item()
    return 2 * num / (den + smooth)


def main():
    m1 = Meter()
    print(m1)
    # m2 = Metric()
    # m2["spe"] = 200
    # m1.update_max(m2)
    # print(m1)
    pass


if __name__ == "__main__":
    main()
