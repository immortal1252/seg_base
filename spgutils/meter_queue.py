class MeterQueue:
    """
    维护capacity个epoch内最好的均值
    """

    def __init__(self, capacity, mode="max"):
        assert mode in ["max", "min"]
        self._queue = []
        self._capacity = capacity
        self._curr_val = None
        self._best_val = None
        self._best_epoch = None
        self._mode = mode

    def append(self, ele, epoch):
        if len(self._queue) < self._capacity:
            self._queue.append(ele)
            self._best_epoch = epoch
            if self._best_val is None:
                self._curr_val = ele
                self._best_val = ele
            else:
                self._best_val += ele
                self._curr_val += ele

        else:
            pop = self._queue.pop(0)
            self._queue.append(ele)
            self._curr_val += ele - pop
            if (self._mode == "max" and self._curr_val > self._best_val) or (
                self._mode == "min" and self._curr_val < self._best_val
            ):
                self._best_val = self._curr_val
                self._best_epoch = epoch

    def get_best_val(self):
        if self._best_val is None:
            return None
        return self._best_val / len(self._queue)

    def get_best_epoch(self):
        if self._best_val is None:
            return None
        return self._best_epoch - self._capacity + 1, self._best_epoch
