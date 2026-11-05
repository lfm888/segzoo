"""LiteSeg: A Novel Lightweight ConvNet for Semantic Segmentation
Paper: https://arxiv.org/abs/1912.06683
Original code: zh320 (2023/10/15)，源自 python_model_train 项目，迁入 segzoo 时的改动见 backbone.py，
以及下面的 _replace_first_conv / encoder 属性 / liteseg() 构建函数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Mobilenetv2, ResNet
from .modules import ConvBNAct, conv1x1


def _replace_first_conv(backbone, in_channels):
    """把骨干网络的第一个卷积换成 in_channels 输入的版本，权重取 RGB 三通道的均值。
    源码只对 ResNet（backbone.conv1）做了这件事，MobileNetV2 没有 conv1 属性会直接报错；
    这里按“第一个 Conv2d”来找，两种骨干都适用。"""
    name, old = next((n, m) for n, m in backbone.named_modules() if isinstance(m, nn.Conv2d))
    parent_name, _, attr = name.rpartition(".")
    parent = backbone.get_submodule(parent_name) if parent_name else backbone

    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(parent, attr, new)


class LiteSeg(nn.Module):
    def __init__(self, num_class=1, n_channel=1, backbone_type="resnet18", act_type="relu", pretrained=False):
        super().__init__()
        if backbone_type == "mobilenet_v2":
            self.backbone = Mobilenetv2(pretrained=pretrained)
            channels = [320, 32]
        elif "resnet" in backbone_type:
            self.backbone = ResNet(backbone_type, pretrained=pretrained)
            channels = [512, 128] if backbone_type in ["resnet18", "resnet34"] else [2048, 512]
        else:
            raise NotImplementedError("LiteSeg 不支持的骨干: %s（可选 resnet18/34/50/101/152 或 mobilenet_v2）"
                                      % backbone_type)

        if n_channel != 3:
            _replace_first_conv(self.backbone, n_channel)

        self.daspp = DASPPModule(channels[0], 512, act_type)
        self.seg_head = SegHead(512 + channels[1], num_class, act_type)

    @property
    def encoder(self):
        """与 smp 模型统一的命名：train.py 的分层学习率（encoder_lr_mult）通过它找到骨干。
        这是普通属性而不是子模块，不会让参数被注册两次。"""
        return self.backbone

    def forward(self, x):
        size = x.size()[2:]

        _, x1, _, x = self.backbone(x)
        size1 = x1.size()[2:]

        x = self.daspp(x)
        x = F.interpolate(x, size1, mode="bilinear", align_corners=True)
        x = torch.cat([x, x1], dim=1)

        x = self.seg_head(x)
        x = F.interpolate(x, size, mode="bilinear", align_corners=True)
        return x


class DASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        hid_channels = in_channels // 5
        last_channels = in_channels - hid_channels * 4
        self.stage1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        self.stage2 = ConvBNAct(in_channels, hid_channels, 3, dilation=3, act_type=act_type)
        self.stage3 = ConvBNAct(in_channels, hid_channels, 3, dilation=6, act_type=act_type)
        self.stage4 = ConvBNAct(in_channels, hid_channels, 3, dilation=9, act_type=act_type)
        self.stage5 = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv1x1(in_channels, last_channels))
        self.conv = ConvBNAct(2 * in_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]

        x1 = self.stage1(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x)
        x5 = F.interpolate(self.stage5(x), size, mode="bilinear", align_corners=True)

        return self.conv(torch.cat([x, x1, x2, x3, x4, x5], dim=1))


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, hid_channels=256):
        super().__init__(
            ConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
            ConvBNAct(hid_channels, hid_channels // 2, 3, act_type=act_type),
            conv1x1(hid_channels // 2, num_class),
        )


def liteseg(num_classes, in_channels=1, backbone="resnet18", act_type="relu", pretrained=False,
            img_size=None):
    """构建 LiteSeg，签名与 sota/__init__.py 的注册表约定一致（img_size 对 LiteSeg 无意义，忽略）。
    预训练权重下载失败时（常见于离线环境）退回随机初始化，并打印警告，与 models.build_model 的行为一致。"""
    try:
        return LiteSeg(num_classes, n_channel=in_channels, backbone_type=backbone,
                       act_type=act_type, pretrained=pretrained)
    except Exception as e:
        if not pretrained:
            raise
        print("[warn] LiteSeg 骨干预训练权重加载失败（%s: %s），改为随机初始化" % (type(e).__name__, e))
        return LiteSeg(num_classes, n_channel=in_channels, backbone_type=backbone,
                       act_type=act_type, pretrained=False)
