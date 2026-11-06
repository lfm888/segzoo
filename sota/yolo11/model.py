"""YOLO11 语义分割（sota/yolo11）：YOLO11 的骨干 + 颈部 + 一个轻量的语义分割头。

  骨干/颈部 = YOLO11 的 Conv / C3k2 / SPPF / C2PSA + FPN-PAN（对应 Ultralytics 配置 yolo11*.yaml 的第 0~22 层）
  分割头     = 本文件自己写的：融合 P3/P4/P5 三个尺度 -> 上采样到 1/4 -> 1x1 分类 -> 双线性放大回输入尺寸
               （不是 Ultralytics 的实例分割头：没有检测框/NMS/mask 系数，输出与其它 sota 模型一样是 (B, num_classes, H, W) logits）

sota_backbone 取值（缩放档位）：n / s / m / l / x；后面加 "-nopsa" 去掉 C2PSA 注意力（如 n-nopsa，给 NPU 部署试验用）。

预训练权重（可选，只加载骨干+颈部，自定义分割头始终随机初始化）：
  - 用 --encoder_weights 指定：Ultralytics 的 yolo11*-seg.pt，或转换后的纯 state_dict（.pth）。
  - --encoder_weights imagenet / true 时自动找 sota/yolo11/{checkpoints,chekcpoints}/yolo11<scale>-seg.pt。
  - Ultralytics 的 .pt 里 pickle 的是模型对象，读取需要 pip install ultralytics；没有该包时先转成纯权重：
        from ultralytics import YOLO; import torch
        torch.save(YOLO("yolo11m-seg.pt").model.state_dict(), "yolo11m-seg.state_dict.pth")
  - 预训练权重用 SiLU 训练，迁移学习建议 --sota_act silu；in_channels=1 时第一层卷积按通道求和（与 smp 一致）。

没有 encoder 属性：train.py 不会做“编码器/解码器分层学习率”，所有参数用同一个学习率。

本文件是对公开的 YOLO11 结构的独立重写，没有复制 Ultralytics 的源码（Ultralytics 代码为 AGPL-3.0）。

自检（不需要数据）：在仓库根目录运行  python -m sota.yolo11.model
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 缩放档位 (depth, width, max_channels)，与 Ultralytics yolo11.yaml 的 scales 一致
SCALES = {"n": (0.50, 0.25, 1024), "s": (0.50, 0.50, 1024), "m": (0.50, 1.00, 512),
          "l": (1.00, 1.00, 512), "x": (1.00, 1.50, 512)}

_ACTS = {"silu": nn.SiLU, "relu": nn.ReLU, "relu6": nn.ReLU6, "hardswish": nn.Hardswish,
         "gelu": nn.GELU, "leakyrelu": lambda: nn.LeakyReLU(0.1)}


def make_act(name):
    key = str(name).lower()
    if key not in _ACTS:
        raise ValueError("不支持的激活函数 %r，可选: %s" % (name, ", ".join(sorted(_ACTS))))
    return _ACTS[key]()


def make_divisible(x, divisor=8):
    return int(math.ceil(x / divisor) * divisor)


# --------------------------------------------------------------------------
# 基本模块（属性名与 Ultralytics 保持一致：conv/bn/act、cv1/cv2/cv3/m、attn/ffn ...，这样权重 key 才能对上）
# --------------------------------------------------------------------------
class Conv(nn.Module):
    """Conv2d(无 bias) + BN + 激活。act=False 表示不加激活。"""

    def __init__(self, c1, c2, k=1, s=1, g=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        # eps=1e-3 与 Ultralytics 训练时一致（加载其 BN 统计量时才不会有系统偏差）；
        # momentum 保持 PyTorch 默认 0.1：小数据集每轮步数少，BN 滑动统计量需要更快跟上
        self.bn = nn.BatchNorm2d(c2, eps=1e-3)
        self.act = make_act(act) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act="silu"):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1, act=act)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    """CSP 块（含 n 个 3x3 Bottleneck）。Ultralytics 里 C3k 继承自 C3，这里合并成一个类。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, act="silu"):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=act)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0, act=act) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    """YOLO11 的主力块：C2f 结构，内部块可选 Bottleneck(c3k=False) 或 C3k(c3k=True)。"""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, act="silu"):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=act)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g, act=act) if c3k else Bottleneck(self.c, self.c, shortcut, g, act=act)
            for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, act="silu"):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """多头自注意力（空间 token）+ 深度卷积位置编码。含 MatMul/Softmax，转 RKNN 前请实测。"""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        h = dim + self.key_dim * num_heads * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, act="silu"):
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1, act=act), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    """骨干最深层（SPPF 之后）的空间注意力模块。"""

    def __init__(self, c1, c2, n=1, e=0.5, act="silu"):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = Conv(2 * self.c, c1, 1, act=act)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64, act=act) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


# --------------------------------------------------------------------------
# 语义分割头（自己设计）：融合 P3/P4/P5 -> 1/4 分辨率 -> 分类 -> 放大回输入尺寸
# --------------------------------------------------------------------------
class SemanticHead(nn.Module):
    def __init__(self, c3, c4, c5, num_classes, act="silu"):
        super().__init__()
        mid = min(max(c3, 64), 128)
        self.l3 = Conv(c3, mid, 1, act=act)
        self.l4 = Conv(c4, mid, 1, act=act)
        self.l5 = Conv(c5, mid, 1, act=act)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=4, mode="nearest")
        self.fuse = Conv(3 * mid, mid, 3, act=act)
        self.up_t = nn.ConvTranspose2d(mid, mid, 2, 2, 0)      # 1/8 -> 1/4（与 YOLO 的 Proto 做法类似）
        self.post = Conv(mid, mid, 3, act=act)
        self.cls = nn.Conv2d(mid, num_classes, 1)

    def forward(self, p3, p4, p5):
        x = self.fuse(torch.cat([self.l3(p3), self.up2(self.l4(p4)), self.up4(self.l5(p5))], 1))
        x = self.cls(self.post(self.up_t(x)))
        # 固定倍数放大（静态 scale，ONNX 导出为定值）；输入尺寸需为 32 的倍数
        return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)


class YOLO11Seg(nn.Module):
    """self.model 是长度 23 的 ModuleList，下标 = Ultralytics yolo11.yaml 的层号（无参数的层用 Identity 占位），
    所以 state_dict 的 key 形如 model.0.conv.weight，与 Ultralytics 权重一致。"""

    def __init__(self, num_classes, in_channels=3, scale="n", act="silu", use_psa=True):
        super().__init__()
        depth, width, max_ch = SCALES[scale]
        ch = lambda c: make_divisible(min(c, max_ch) * width, 8)          # 通道缩放
        rep = lambda n: max(round(n * depth), 1) if n > 1 else n          # 重复次数缩放
        k_all = scale in ("m", "l", "x")                                  # m/l/x 档所有 C3k2 都用 C3k
        a, b, c, d, e = ch(64), ch(128), ch(256), ch(512), ch(1024)
        n = rep(2)

        L = [nn.Identity() for _ in range(23)]
        L[0] = Conv(in_channels, a, 3, 2, act=act)                         # P1/2
        L[1] = Conv(a, b, 3, 2, act=act)                                   # P2/4
        L[2] = C3k2(b, c, n, k_all, 0.25, act=act)
        L[3] = Conv(c, c, 3, 2, act=act)                                   # P3/8
        L[4] = C3k2(c, d, n, k_all, 0.25, act=act)
        L[5] = Conv(d, d, 3, 2, act=act)                                   # P4/16
        L[6] = C3k2(d, d, n, True, act=act)
        L[7] = Conv(d, e, 3, 2, act=act)                                   # P5/32
        L[8] = C3k2(e, e, n, True, act=act)
        L[9] = SPPF(e, e, 5, act=act)
        if use_psa:
            L[10] = C2PSA(e, e, n, act=act)
        # ---- 颈部：FPN（自顶向下）+ PAN（自底向上）----
        L[13] = C3k2(e + d, d, n, k_all, act=act)
        L[16] = C3k2(d + d, c, n, k_all, act=act)                          # 输出 P3（1/8）
        L[17] = Conv(c, c, 3, 2, act=act)
        L[19] = C3k2(c + d, d, n, k_all, act=act)                          # 输出 P4（1/16）
        L[20] = Conv(d, d, 3, 2, act=act)
        L[22] = C3k2(d + e, e, n, True, act=act)                           # 输出 P5（1/32）
        self.model = nn.ModuleList(L)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = SemanticHead(c, d, e, num_classes, act=act)

    def forward(self, x):
        m = self.model
        x = m[3](m[2](m[1](m[0](x))))
        p3b = m[4](x)                                   # 骨干 P3（1/8）
        p4b = m[6](m[5](p3b))                           # 骨干 P4（1/16）
        p5b = m[10](m[9](m[8](m[7](p4b))))              # 骨干 P5（1/32，经 SPPF + C2PSA）
        h13 = m[13](torch.cat([self.up(p5b), p4b], 1))
        p3 = m[16](torch.cat([self.up(h13), p3b], 1))
        p4 = m[19](torch.cat([m[17](p3), h13], 1))
        p5 = m[22](torch.cat([m[20](p4), p5b], 1))
        return self.head(p3, p4, p5)


# --------------------------------------------------------------------------
# 预训练权重（Ultralytics yolo11*-seg.pt / 纯 state_dict）
# --------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]   # segzoo 仓库根目录


def _find_default_ckpt(scale):
    """按缩放档位找默认权重：sota/yolo11/checkpoints/yolo11<scale>-seg.pt（兼容 chekcpoints 拼写）。"""
    here = Path(__file__).resolve().parent
    for sub in ("checkpoints", "chekcpoints", "weights"):
        for name in ("yolo11%s-seg.pt" % scale, "yolo11%s.pt" % scale):
            p = here / sub / name
            if p.is_file():
                return p
    return None


def _resolve_ckpt(value, scale):
    """把 --encoder_weights 的值解析成权重文件路径；None = 从零训练。"""
    if value is False or value is None:
        return None
    v = str(value).strip()
    if not v or v.lower() in ("none", "false", "0"):
        return None
    if v.lower() in ("true", "imagenet", "1"):
        p = _find_default_ckpt(scale)
        if p is None:
            print("[yolo11] 没找到 sota/yolo11/checkpoints/yolo11%s-seg.pt，改为从零训练" % scale)
        return p
    p = Path(v).expanduser()
    if not p.is_absolute():
        cand = _REPO_ROOT / p
        p = cand if cand.is_file() else p
    if not p.is_file():
        raise FileNotFoundError("--encoder_weights 指向的权重文件不存在: %s" % p)
    return p


def _extract_state_dict(ckpt):
    """从 Ultralytics 检查点（含 model/ema 模型对象）、纯 state_dict、或本项目 best.pth 里取出权重字典。"""
    if not isinstance(ckpt, dict):
        return ckpt
    sd = ckpt.get("state_dict")
    if isinstance(sd, dict):
        return sd
    for key in ("ema", "model"):
        obj = ckpt.get(key)
        if obj is None:
            continue
        if isinstance(obj, dict):            # 本项目 best.pth / 纯 state_dict
            return obj
        if hasattr(obj, "state_dict"):       # Ultralytics 的 SegmentationModel 对象
            return obj.state_dict()
    return ckpt


class _StubMeta(type):
    def __getattr__(cls, name):              # 反序列化时允许访问任意的类属性（如 Detect.forward）
        return None


class _StubModule(nn.Module, metaclass=_StubMeta):
    """空壳模块：只承接 pickle 里的结构/参数，不执行 Ultralytics 的代码。"""


def _load_ultralytics_without_pkg(path):
    """没装 ultralytics 时读取其 .pt：把 ultralytics.* 的类替换成空壳再反序列化，只取权重。"""
    import pickle
    import types

    class _StubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("ultralytics"):
                return _StubMeta(name, (_StubModule,), {})
            return super().find_class(module, name)

    mod = types.SimpleNamespace(Unpickler=_StubUnpickler, __name__="yolo11_stub_unpickler")
    return torch.load(str(path), map_location="cpu", weights_only=False, pickle_module=mod)


def _load_any_ckpt(path):
    """读权重文件：先按纯 tensor 读（weights_only=True），失败再按完整 pickle 读；
    没装 ultralytics 时最后用兼容模式（空壳反序列化）读 Ultralytics 的 .pt。"""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        pass
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        try:
            ckpt = _load_ultralytics_without_pkg(path)
            print("[yolo11] 未安装 ultralytics，已用兼容模式读取 %s（只取权重）" % Path(path).name)
            return ckpt
        except Exception as e2:
            raise RuntimeError(
                "读取 %s 失败（%s / %s）。可 pip install ultralytics 后重试，或先转成纯 state_dict：\n"
                "    from ultralytics import YOLO; import torch\n"
                "    torch.save(YOLO('%s').model.state_dict(), '%s')"
                % (path, e, e2, Path(path).name, Path(path).with_suffix(".state_dict.pth").name)
            ) from e2


def _load_pretrained(model, ckpt_path, in_channels, act_type):
    """加载形状匹配的权重（Ultralytics 的骨干+颈部 model.0~22；它自带的 model.23 检测/分割头不匹配会被跳过）。
    返回 (加载的 tensor 数, 跳过的 tensor 数)。"""
    ckpt = _load_any_ckpt(ckpt_path)
    sd = _extract_state_dict(ckpt)
    if not isinstance(sd, dict):
        raise RuntimeError("无法从 %s 解析出 state_dict" % ckpt_path)

    w0 = sd.get("model.0.conv.weight")
    if w0 is not None and w0.ndim == 4 and w0.shape[1] != int(in_channels):
        if int(in_channels) == 1:
            sd["model.0.conv.weight"] = w0.sum(dim=1, keepdim=True)     # 与 smp / liteseg 的单通道做法一致
        else:
            sd["model.0.conv.weight"] = w0.mean(dim=1, keepdim=True).repeat(1, int(in_channels), 1, 1)

    own = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if isinstance(v, torch.Tensor) and k in own and own[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    if str(act_type).lower() != "silu":
        print("[yolo11] 提示：预训练权重用 SiLU 训练，当前 sota_act=%r；迁移学习建议 --sota_act silu" % act_type)
    return len(matched), len(sd) - len(matched)


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def yolo11(num_classes, in_channels=1, backbone="n", act_type="relu", pretrained=False, img_size=None):
    """sota 统一入口。backbone = 缩放档位 n/s/m/l/x，可加 -nopsa 去掉 C2PSA 注意力。输出未经 softmax 的 logits (B, C, H, W)。

    pretrained: False / None / "none" = 从零训练；
                True / "imagenet"     = 自动找 sota/yolo11/checkpoints/yolo11<scale>-seg.pt；
                其它字符串             = 权重文件路径（Ultralytics 的 .pt 或转换后的纯 state_dict）。"""
    scale, _, tag = str(backbone).partition("-")
    if scale not in SCALES or tag not in ("", "nopsa"):
        raise ValueError("yolo11 的 sota_backbone 只能是 %s（可加 -nopsa），当前 %r" % ("/".join(SCALES), backbone))
    model = YOLO11Seg(num_classes, in_channels, scale, act_type, use_psa=(tag != "nopsa"))
    ckpt = _resolve_ckpt(pretrained, scale)
    if ckpt is not None:
        n, skipped = _load_pretrained(model, ckpt, in_channels, act_type)
        print("[yolo11] 已从 %s 加载骨干+颈部 %d 个 tensor（跳过 %d 个；自定义分割头随机初始化）"
              % (ckpt.name, n, skipped))
    return model


# --------------------------------------------------------------------------
# 自检：python -m sota.yolo11.model
# --------------------------------------------------------------------------
DEFAULTS = {"scale": "n", "in_channels": 3, "num_classes": 3, "img_size": [128, 128], "act": "relu", "onnx_check": True}

# yolo11n 各层参数量参考值（Ultralytics 训练日志里的数字，凭记忆整理；不一致时以官方日志为准）
_REF_N = {0: 464, 1: 4672, 2: 6640, 3: 36992, 4: 26080, 5: 147712, 6: 87040, 7: 295424, 8: 346112, 9: 164608,
          10: 249728, 13: 111296, 16: 32096, 17: 36992, 19: 86720, 20: 147712, 22: 378880}


def _selftest():
    from utils import parse_args
    cfg = parse_args(DEFAULTS, "yolo11 自检")
    H, W = cfg["img_size"]
    m = yolo11(cfg["num_classes"], cfg["in_channels"], cfg["scale"], cfg["act"], pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.randn(1, cfg["in_channels"], H, W))
    assert tuple(y.shape) == (1, cfg["num_classes"], H, W), y.shape
    print("前向 OK：输出", tuple(y.shape))
    total = 0
    for i, layer in enumerate(m.model):
        p = sum(t.numel() for t in layer.parameters())
        total += p
        if p:
            ref = _REF_N.get(i) if (cfg["scale"] == "n" and cfg["in_channels"] == 3) else None
            print("  层 %2d %-8s %8d%s" % (i, type(layer).__name__, p,
                                          "" if ref is None else ("   参考 %d %s" % (ref, "OK" if ref == p else "不一致"))))
    head = sum(t.numel() for t in m.head.parameters())
    print("骨干+颈部 %.3f M，分割头 %.3f M，合计 %.3f M" % (total / 1e6, head / 1e6, (total + head) / 1e6))
    if cfg["onnx_check"]:
        import io
        buf = io.BytesIO()
        torch.onnx.export(m, torch.randn(1, cfg["in_channels"], H, W), buf, opset_version=12,
                          input_names=["input"], output_names=["logits"])
        print("ONNX(opset 12) 导出 OK：%.1f MB" % (len(buf.getvalue()) / 1e6))


if __name__ == "__main__":
    _selftest()