"""LiteSeg（轻量分割网络）。训练时通过 train.py 的 sota_model=liteseg 启用，也可以直接：

    from sota.liteseg import liteseg
    model = liteseg(num_classes=3, in_channels=1, backbone="resnet18")
"""
from .model import LiteSeg, liteseg

__all__ = ["LiteSeg", "liteseg"]
