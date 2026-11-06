"""SOTA 模型（不属于 smp 的网络：LiteSeg、TransUNet ...）的统一入口。

使用方法（train.py 的参数）：
    sota_model: liteseg          # 模型名，见 SOTA_INFO
    sota_backbone: resnet18      # 骨干：SOTA_INFO 里 backbones 为空的模型（固定结构）必须留空
    sota_act: relu               # 激活函数：uses_act 为 False 的模型忽略

新增模型：
    1. 在 sota/<模型名>/ 下实现构建函数，签名为
       build(num_classes, in_channels, backbone, act_type, pretrained, img_size) -> nn.Module
       输出为未经 softmax 的 logits，形状 (B, num_classes, H, W)。
    2. 在下面 SOTA_MODELS 和 SOTA_INFO 各登记一项。
"""
from .liteseg import liteseg
from .transunet import transunet
from .yolo11 import yolo11

SOTA_MODELS = {
    "liteseg": liteseg,
    "transunet": transunet,
    "yolo11": yolo11,          # 【新增】登记 YOLO11
}

# backbones 为空 = 结构固定，没有可选骨干（sota_backbone 必须留空）
# square / divisor：输入的约束（TransUNet 的解码器只支持正方形、16 的倍数）
SOTA_INFO = {
    "liteseg": dict(desc="LiteSeg：DASPP + 轻量解码器",
                    backbones=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "mobilenet_v2"],
                    uses_act=True, square=False, divisor=None),
    "transunet": dict(desc="TransUNet：R50 + ViT-B/16",
                      backbones=[], uses_act=False, square=True, divisor=16),
    # 【新增】YOLO11 的配置说明与合法骨干（档位）
    "yolo11": dict(desc="YOLO11 Semantic Segmentation",
                   backbones=["n", "s", "m", "l", "x", "n-nopsa", "s-nopsa", "m-nopsa", "l-nopsa", "x-nopsa"],
                   uses_act=True, square=False, divisor=32),
}


def sota_names():
    return sorted(SOTA_MODELS)


def sota_info(name):
    key = str(name).strip().lower()
    if key not in SOTA_MODELS:
        raise ValueError("sota_model=%r 不存在，可选: %s（在 sota/__init__.py 登记）"
                         % (name, ", ".join(sota_names())))
    return SOTA_INFO[key]


def build_sota(name, num_classes, in_channels=1, backbone="", act_type="relu", pretrained=False, img_size=None):
    """按名字构建 SOTA 模型。backbone 对固定结构的模型会被忽略。"""
    key = str(name).strip().lower()
    sota_info(key)                       # 不存在时报错
    return SOTA_MODELS[key](num_classes, in_channels=in_channels, backbone=backbone,
                            act_type=act_type, pretrained=pretrained, img_size=img_size)
