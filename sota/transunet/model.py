"""TransUNet（R50 + ViT-B/16）构建函数。

结构是固定的：ResNet-50 作为混合 stem，接 ViT-B/16 编码器，再接 U-Net 式解码器。
没有可选的骨干，所以 sota_backbone 必须留空（见 sota/__init__.py 的 SOTA_INFO）。

源码：Beckschen/TransUNet（Apache-2.0，见 LICENSE）。vit_seg_*.py 保持上游文件名与结构，便于以后同步上游。

迁入时处理的几个问题：
  1. 上游的 get_b16_config() 缺少 n_skip / skip_channels，只能用 get_r50_b16_config()。
  2. config.patches.grid 必须跟随 img_size 设置为 (img_size//16, img_size//16)，否则 patch 大小算成 0。
  3. 解码器用 int(np.sqrt(n_patch)) 还原特征图，所以只支持正方形输入，且 img_size 必须是 16 的倍数。
  4. 上游用 os.path.join 拼 .npz 权重的 key，在 Windows 下会出错。已在 vit_seg_*.py 里改为 '/'。
  5. 预训练权重默认放在 sota/transunet/checkpoints/R50+ViT-B_16.npz（约 380MB，不随代码分发，需要自行下载）。
"""
from pathlib import Path

import numpy as np

from .vit_seg_configs import get_r50_b16_config
from .vit_seg_modeling import VisionTransformer

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "checkpoints" / "R50+ViT-B_16.npz"


class TransUNet(VisionTransformer):
    @property
    def encoder(self):
        """与其它模型统一的命名：train.py 的分层学习率通过它找到预训练的 ViT 编码器（含 ResNet stem）。
        普通属性而不是子模块，不会让参数被注册两次。"""
        return self.transformer


def transunet(num_classes, in_channels=1, backbone=None, act_type=None, pretrained=False,
              img_size=224, weights_path=None):
    """构建 TransUNet。backbone / act_type 为统一签名保留，结构固定，传入会被忽略。

    Args:
        num_classes: 输出通道数（含背景）。
        in_channels: 1 或 3。单通道在 forward 里会自动复制成 3 通道。
        img_size:    正方形输入边长，必须是 16 的倍数。
        pretrained:  True 时加载 weights_path（默认 DEFAULT_WEIGHTS）。文件不存在则随机初始化并警告。
    """
    if in_channels not in (1, 3):
        raise ValueError("TransUNet 只支持 1 或 3 通道输入，当前 in_channels=%s" % in_channels)
    if img_size is None:
        raise ValueError("TransUNet 需要指定 img_size（正方形边长，16 的倍数）")
    img_size = int(img_size)
    if img_size % 16 != 0:
        raise ValueError("TransUNet 要求 img_size 是 16 的倍数，当前 %d" % img_size)

    cfg = get_r50_b16_config()
    cfg.n_classes = int(num_classes)
    cfg.n_skip = 3
    cfg.patches.grid = (img_size // 16, img_size // 16)   # 必须跟随 img_size，否则 patch 大小为 0

    model = TransUNet(cfg, img_size=img_size, num_classes=int(num_classes))

    if pretrained:
        path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        if not path.is_file():
            print("[warn] TransUNet 预训练权重不存在: %s，从零开始训练。"
                  "下载 R50+ViT-B_16.npz 后放到该位置即可（见 docs/sota.md）" % path)
        else:
            try:
                model.load_from(np.load(str(path), allow_pickle=True))
            except Exception as e:
                print("[warn] TransUNet 预训练权重加载失败（%s: %s），改为随机初始化" % (type(e).__name__, e))
    return model
