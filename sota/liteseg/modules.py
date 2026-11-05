"""LiteSeg 用到的卷积积木（源项目 modules.py 的精简版，只保留 LiteSeg 实际调用的部分）。"""
import torch.nn as nn


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    """1x1 卷积（逐点卷积）"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)


class ConvBNAct(nn.Sequential):
    """卷积 -> BatchNorm -> 激活"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=False, act_type="relu", **kwargs):
        if isinstance(kernel_size, (list, tuple)):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        else:
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs),
        )


class Activation(nn.Module):
    """按名字选择激活函数，名字大小写不敏感。"""

    _HUB = {
        "relu": nn.ReLU, "relu6": nn.ReLU6, "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU,
        "celu": nn.CELU, "elu": nn.ELU, "hardswish": nn.Hardswish, "hardtanh": nn.Hardtanh,
        "gelu": nn.GELU, "selu": nn.SELU, "silu": nn.SiLU, "tanh": nn.Tanh, "none": nn.Identity,
    }

    def __init__(self, act_type, **kwargs):
        super().__init__()
        key = str(act_type).lower()
        if key not in self._HUB:
            raise NotImplementedError("不支持的激活函数: %s（可选 %s）" % (act_type, ", ".join(self._HUB)))
        self.activation = self._HUB[key](**kwargs)

    def forward(self, x):
        return self.activation(x)
