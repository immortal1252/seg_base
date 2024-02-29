import math


class Meter:
    def __init__(self, pre=4):
        super().__init__()
        self._data = {
            "dice": None,
            "iou": None,
            "hd95": None,
            "pre": None,
            "rec": None,
            "spe": None,
        }
        # 添加的时候自动累加，便于求平均值
        self._len = 0
        self._pre = pre

    def __getitem__(self, key):
        assert key in self._data.keys(), "bad key"
        return self._data[key]

    def __setitem__(self, key, value):
        assert key in self._data.keys(), "bad key"
        self._data[key] = value

    def __iadd__(self, other):
        self._len += 1
        for key in self._data.keys():
            if self._data[key] is None:
                self._data[key] = other._data[key]
            elif other._data[key] is not None:
                self._data[key] += other._data[key]

        return self

    def mean(self):
        # 返回平均值，为了防止多次调用出问题，返回副本
        ret = Meter()
        ret._len = 1
        ret._data = self._data.copy()
        for key in ret._data.keys():
            if ret._data[key] is not None:
                ret._data[key] /= self._len

        return ret

    def update_max(self, other):
        self._len = 1
        for key in self._data.keys():
            if self._data[key] is None:
                self._data[key] = other[key]
            elif key != "hd95":
                self._data[key] = max(self._data[key], other[key])
            else:
                self._data[key] = min(self._data[key], other[key])

    def __str__(self):
        map_float2 = self._data.copy()
        for key in map_float2.keys():
            value = map_float2[key]
            if isinstance(value, float) and not math.isnan(value):
                # 保留两位小数
                newvalue = round(value, self._pre)
                map_float2[key] = newvalue

        return map_float2.__str__()


def main():
    m1 = Meter()
    m1["dice"] = 1 / 3
    print(m1)
    # m2 = Metric()
    # m2["spe"] = 200
    # m1.update_max(m2)
    # print(m1)
    pass


if __name__ == "__main__":
    main()
