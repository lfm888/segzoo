"""LiteSeg 的骨干网络：ResNet（torchvision）与 MobileNetV2。

两者的 forward 都返回 4 个尺度的特征 (x1, x2, x3, x4)，下采样倍数分别为 4 / 8 / 16 / 32。
LiteSeg 只用 x2（8x）和 x4（32x）。

相对源项目（python_model_train/models/backbone.py）的改动只有两处，骨干的结构与源码一致：
  1. 预训练权重改用 torchvision 新版的 weights= 参数（旧的 pretrained= 已弃用）。
  2. 去掉了源文件开头的 UTF-8 BOM。
（MobileNetV2 的分段 features[14:18] 与源码相同，输出 320 通道，正好是 LiteSeg 的 channels=[320, 32]；
 features[18] 才是扩展到 1280 的 1x1 卷积，不属于骨干。）
MobileNet 单通道输入的首层卷积问题在 model.py 的 _replace_first_conv 中修复。
"""
import torch.nn as nn


class ResNet(nn.Module):
    """torchvision ResNet 的四个 stage。pretrained=True 时下载 ImageNet 权重。"""

    def __init__(self, resnet_type, pretrained=False):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

        hub = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
               "resnet101": resnet101, "resnet152": resnet152}
        if resnet_type not in hub:
            raise ValueError("不支持的 ResNet 类型: %s（可选 %s）" % (resnet_type, ", ".join(hub)))

        net = hub[resnet_type](weights="IMAGENET1K_V1" if pretrained else None)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))   # 4x
        x1 = self.layer1(x)     # 4x
        x2 = self.layer2(x1)    # 8x
        x3 = self.layer3(x2)    # 16x
        x4 = self.layer4(x3)    # 32x
        return x1, x2, x3, x4


class Mobilenetv2(nn.Module):
    """torchvision MobileNetV2 的 features 按下采样位置切成四段。"""

    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        net = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        f = net.features
        self.layer1 = f[:4]      # 4x,  24 通道
        self.layer2 = f[4:7]     # 8x,  32 通道
        self.layer3 = f[7:14]    # 16x, 96 通道
        self.layer4 = f[14:18]   # 32x, 320 通道

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4
