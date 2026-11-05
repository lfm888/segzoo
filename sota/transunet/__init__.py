"""TransUNet（R50 + ViT-B/16，固定结构）。训练时通过 sota_model=transunet 启用，sota_backbone 留空。

    from sota.transunet import transunet
    model = transunet(num_classes=3, in_channels=1, img_size=224)
"""
from .model import TransUNet, transunet

__all__ = ["TransUNet", "transunet"]
