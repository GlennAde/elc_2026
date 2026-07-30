import numpy as np


class Calibration:
    """像素→厘米映射

    支持两种模式:
    1. 多点标定: from_file() 加载 .npz 文件, np.interp 插值
    2. 线性降级: from_linear() 仅用两端点, 等价旧 scale_k 行为
    """

    def __init__(self, px, cm):
        self.px = np.array(px, dtype=np.float32)
        self.cm = np.array(cm, dtype=np.float32)

    @classmethod
    def from_file(cls, path):
        data = np.load(path)
        return cls(data["px"], data["cm"])

    @classmethod
    def from_linear(cls, pixel_left, pixel_right, bar_length_cm):
        px = np.array([pixel_left, pixel_right], dtype=np.float32)
        cm = np.array([-bar_length_cm / 2.0, bar_length_cm / 2.0], dtype=np.float32)
        return cls(px, cm)

    def save(self, path):
        np.savez(path, px=self.px, cm=self.cm)

    def px_to_cm(self, x):
        return float(np.interp(x, self.px, self.cm))

    def px_to_cm_clamped(self, x):
        return max(self.cm[0], min(self.cm[-1], self.px_to_cm(x)))

    @property
    def cm_range(self):
        return float(self.cm[0]), float(self.cm[-1])
