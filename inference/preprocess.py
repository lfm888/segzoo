"""推理预处理（pth / onnxruntime / OpenCV DNN 共用），只依赖 numpy + cv2，方便逐行翻译成 C++。

约定：输入是 OpenCV 原生的 BGR uint8（cv2.imread / 摄像头帧）或灰度 uint8；输出是 float32 的 NCHW blob。
归一化 (x/255 - mean)/std 在这里（模型外部）完成，ONNX 里不含归一化。

【统一量纲约定（本文件的核心）】
  * 像素：先检查数值范围，再统一缩放到 [0, 1]（uint8 直接 /255；float 输入必须是 0~255 量纲）。
  * mean / std：全仓一律是 [0, 1] 量纲（ImageNet 即 0.485/0.456/0.406 与 0.229/0.224/0.225）。
  * 归一化只有一个入口 normalize()：  y = (x/255 - mean) / std
    训练（数据集、可视化）、验证、导出、部署都必须经过它，量纲只在这一个地方被处理。
  * 需要 0~255 量纲的下游（如 RKNN config 的 mean_values/std_values）用 spec.mean_255() / spec.std_255() 取，
    不要自己手写乘 255。

归一化方案只有两种（config 里的 norm）：
  "imagenet"  mean=(0.485,0.456,0.406) std=(0.229,0.224,0.225)；灰度取三通道均值作近似
  "01"        mean=0, std=1，即只做 x/255，结果范围 [0,1]

流水线（每一步在 C++ 里都有同名的 OpenCV 调用，Python 与 C++ 的数值定义完全一致）：

  步骤  Python（本文件）                            C++ (OpenCV)
  ----  ----------------------------------------  ----------------------------------------------------
  1     read_image   cv2.imdecode(IMREAD_COLOR|GRAYSCALE)   cv::imread(path, cv::IMREAD_COLOR|IMREAD_GRAYSCALE)
  2     to_model_color  BGR->RGB / BGR->GRAY               cv::cvtColor(img, img, cv::COLOR_BGR2RGB / COLOR_BGR2GRAY)
  3     resize       任一边要缩小 INTER_AREA，否则 INTER_LINEAR   cv::resize(img, img, cv::Size(W,H), 0, 0, interp)
  4a    to_unit_range 检查像素范围，uint8 -> float32 并 /255    img.convertTo(f, CV_32F, 1.0/255.0);
  4b    normalize    (x - mean) / std（逐通道，0~1 量纲）      cv::subtract(f, cv::Scalar(mean...), f);
                                                             cv::divide(f, cv::Scalar(std...), f);
  5     to_blob      HWC -> 1xCxHxW，float32，连续内存          cv::dnn::blobFromImage(f, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F)

注意：
  * 第 4 步不能用 blobFromImage 的 mean/scalefactor 代替：scalefactor 只能是标量，而 3 通道的 std 逐通道不同。
    所以 C++ 里用 cv::subtract / cv::divide + Scalar 逐通道处理（mean/std 通道顺序 = RGB 顺序，因为第 2 步已经转成 RGB），
    最后的 blobFromImage 只做 HWC->NCHW 的排布变换（swapRB=false，mean 为空，scalefactor=1）。
  * 第 3 步的规则与训练完全一致：只要宽或高有一边需要缩小，两个方向都用 INTER_AREA，
    否则用 INTER_LINEAR；尺寸本来就一样时不做缩放。
  * 后处理见文件末尾：argmax 取“第一个最大值”（与 np.argmax 并列时一致），再最近邻放大回原图。
  * 旧版 model_meta.json / best.pth 里的 mean/std 是 0~255 量纲（如 123.675），from_meta 会自动识别并换算成 0~1，
    并给出一条警告；新训练的 meta 带有 "norm_units": "0-1" 标记。

直接运行本文件 = 自检 + 生成 golden 向量（给 C++ 对拍用）：
  python inference/preprocess.py
  python inference/preprocess.py --image data/xxx/train_dataset/001.png --meta runs/xxx/model_meta.json
"""
import argparse
import glob
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent  # segzoo 仓库根目录（inference/ 的上一级）

# ============================ 归一化预设（0~1 量纲，通道顺序 RGB） ============================
IMAGENET_MEAN = (0.485, 0.456, 0.406)    # = (123.675, 116.28, 103.53) / 255
IMAGENET_STD = (0.229, 0.224, 0.225)     # = (58.395, 57.12, 57.375) / 255
NORM_CHOICES = ("imagenet", "01")        # config 里 norm 的全部可选值
_INV255 = np.float32(1.0 / 255.0)        # 与 C++ convertTo(f, CV_32F, 1.0/255.0) 的系数对应


def norm_stats(norm, in_channels):
    """按方案名返回 (mean, std) 列表（0~1 量纲，长度 == in_channels）。
    imagenet：3 通道用完整三元组，灰度取三通道均值/标准差作近似
              （与 smp 把预训练 conv1 的三通道权重求和用于单通道输入一致）。
    01      ：mean=0, std=1，即只做 x/255。"""
    c = int(in_channels)
    if c not in (1, 3):
        raise ValueError("in_channels 只支持 1 或 3，当前 %r" % (in_channels,))
    key = str(norm).strip().lower()
    if key == "imagenet":
        if c == 3:
            return list(IMAGENET_MEAN), list(IMAGENET_STD)
        return [sum(IMAGENET_MEAN) / 3.0], [sum(IMAGENET_STD) / 3.0]
    if key == "01":
        return [0.0] * c, [1.0] * c
    raise ValueError("norm 只能是 %s 之一，当前 %r" % (" / ".join(NORM_CHOICES), norm))


# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "meta": "latest",                 # model_meta.json / best.pth 同目录导出的元数据；latest = runs 下最新的 model_meta.json
    "runs_dir": "runs",
    "image": "",                      # 自检用的图像；空 = 用程序合成的图
    "out_dir": "inference/golden",    # golden 向量输出目录（相对仓库根目录）
    "save_golden": True,              # 是否保存 golden 向量
    "check_against_training": True,   # 与训练端旧版 utils.load_image/resize_image/normalize 逐元素对比（需要在仓库里运行）
    # 下面三项仅在找不到 meta 时作为备用（没训练过模型也能先跑自检）
    "norm": "imagenet",               # imagenet / 01
    "in_channels": 1,
    "img_size": [288, 512],           # [H, W]
}
# ===========================================================================


# --------------------------------------------------------------------------
# 预处理规格：一份“契约”，训练端写进 meta / model_meta.json，推理端只读它
# --------------------------------------------------------------------------
def _expand(values, c, name):
    """长度 1 的 mean/std 在这里一次性显式广播到 c 个通道；长度既不是 1 也不是 c 就报错（不再静默处理）。"""
    values = tuple(float(v) for v in values)
    if len(values) == 1:
        return values * c
    if len(values) != c:
        raise ValueError("%s 的长度应为 1 或 %d（通道数），实际 %d: %s" % (name, c, len(values), values))
    return values


@dataclass
class PreprocessSpec:
    in_channels: int            # 1=灰度，3=RGB
    img_size: Tuple[int, int]   # (H, W)，模型输入尺寸
    mean: Tuple[float, ...]     # 0~1 量纲（像素先 /255 再减），长度 == in_channels，通道顺序 = RGB
    std: Tuple[float, ...]      # 0~1 量纲
    norm: str = "custom"        # 方案名（imagenet / 01 / custom），仅作记录

    def __post_init__(self):
        c = int(self.in_channels)
        if c not in (1, 3):
            raise ValueError("in_channels 只支持 1 或 3，当前 %r" % (self.in_channels,))
        h, w = (int(v) for v in self.img_size)
        if h <= 0 or w <= 0:
            raise ValueError("img_size 必须为正数，当前 %r" % (self.img_size,))
        mean, std = _expand(self.mean, c, "mean"), _expand(self.std, c, "std")
        if any(s <= 0 for s in std):
            raise ValueError("std 必须全部 > 0，当前 %s" % (std,))
        # 量纲守卫：mean/std 必须是 0~1 量纲。像 123.675 / 58.395 这样的 0~255 量纲在这里直接拒绝，
        # 否则输入被 /255 以后再减 123.675，整张图会变成接近常数，而且不会报任何错
        if any(m < 0 or m > 1 for m in mean) or any(s > 1 for s in std):
            raise ValueError("mean/std 必须是 0~1 量纲（ImageNet 为 0.485/0.229 这类），当前 mean=%s std=%s；"
                             "如果手里是 0~255 量纲的值请先除以 255" % (mean, std))
        self.in_channels, self.img_size, self.mean, self.std = c, (h, w), mean, std

    # ---------------- 构造 ----------------
    @classmethod
    def from_config(cls, cfg):
        """训练端入口：从 config（DEFAULTS 字典）里的 norm / in_channels / img_size 构造。"""
        c = int(cfg["in_channels"])
        norm = str(cfg.get("norm", "imagenet")).strip().lower()
        mean, std = norm_stats(norm, c)
        return cls(in_channels=c, img_size=cfg["img_size"], mean=mean, std=std, norm=norm)

    @classmethod
    def from_meta(cls, meta):
        """从 train.py 写入 checkpoint 的 meta，或 export_onnx.py 写出的 model_meta.json 构造。
        如果 meta 里有 "preprocess" 子字典就优先用它，否则读顶层的 in_channels/img_size/mean/std。
        兼容旧 meta：没有 norm_units 且数值明显是 0~255 量纲（如 123.675）时，自动除以 255 并警告。"""
        m = meta.get("preprocess") or meta
        mean, std = [float(v) for v in m["mean"]], [float(v) for v in m["std"]]
        units = m.get("norm_units")
        if units is None:                                   # 旧 meta：没有量纲标记
            units = "0-255" if max(max(mean), max(std)) > 1.0 else "0-1"
            if units == "0-255":
                warnings.warn("meta 里的 mean/std 是旧版 0~255 量纲，已自动 /255 换算为 0~1；"
                              "建议用新版 train.py 重新训练/导出以写入 norm_units 标记")
        if units == "0-255":
            mean, std = [v / 255.0 for v in mean], [v / 255.0 for v in std]
        elif units != "0-1":
            raise ValueError("未知的 norm_units: %r（只支持 0-1 / 0-255）" % (units,))
        norm = str(m.get("norm", "custom"))
        return cls(in_channels=m["in_channels"], img_size=m["img_size"], mean=mean, std=std, norm=norm)

    # ---------------- 给 0~255 量纲的下游使用（如 RKNN config） ----------------
    def mean_255(self):
        """mean × 255，供 RKNN config(mean_values=...) 或旧代码使用。"""
        return [v * 255.0 for v in self.mean]

    def std_255(self):
        """std × 255，供 RKNN config(std_values=...) 或旧代码使用。"""
        return [v * 255.0 for v in self.std]

    def to_dict(self):
        return {"in_channels": self.in_channels, "img_size": list(self.img_size),
                "mean": list(self.mean), "std": list(self.std), "norm": self.norm, "norm_units": "0-1",
                "mean_255": self.mean_255(), "std_255": self.std_255(),
                "color_order": "RGB" if self.in_channels == 3 else "GRAY",
                "resize": "INTER_AREA if (src_h > H or src_w > W) else INTER_LINEAR; skip if same size",
                "normalize": "x_f32 = x_u8 / 255 (range-checked, -> [0,1]); y = (x_f32 - mean) / std, per channel, "
                             "mean/std in 0~1 units",
                "layout": "NCHW float32"}


# --------------------------------------------------------------------------
# 预处理的五个步骤（训练端的数据集也应调用这些函数，在步骤 3 与 4 之间插入数据增强）
# --------------------------------------------------------------------------
def read_image(path, in_channels):
    """步骤 1：读文件 -> uint8，3 通道时是 BGR（OpenCV 原生），1 通道时是灰度 HxW。
    用 fromfile + imdecode 是为了 Windows 下支持中文路径；C++ 直接 cv::imread 即可，解码器是同一个。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if int(in_channels) == 1 else cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("读取图像失败: %s" % path)
    return img


def to_model_color(img, in_channels):
    """步骤 2：统一成模型需要的通道形式。输入可以是 BGR(A) / 灰度，输出是 RGB HxWx3 或灰度 HxW。
    - 模型 3 通道：BGR->RGB（灰度输入则复制成 3 通道）
    - 模型 1 通道：BGR->GRAY（已经是灰度则原样返回）"""
    if int(in_channels) == 1:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)


def resize(img, hw):
    """步骤 3：缩放到模型输入尺寸 (H, W)。规则（C++ 必须照抄）：
    宽或高任意一边要缩小 -> 两个方向都用 INTER_AREA；否则用 INTER_LINEAR；尺寸相同则不处理。"""
    h, w = hw
    if img.shape[0] == h and img.shape[1] == w:
        return img
    shrink = img.shape[0] > h or img.shape[1] > w
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def to_unit_range(img):
    """步骤 4a：检查像素数值范围，并统一缩放到 [0, 1] 的 float32（形状不变）。
    - uint8：天然是 0~255，直接 x/255
    - float ：约定是 0~255 量纲（例如数据增强之后的结果）；含 NaN/Inf、或超出 [-0.5, 255.5] 就报错
              （最常见的原因是传进来的已经是 0~1 或者已经归一化过的数据，再 /255 会悄悄把图变成近似全黑）；
              落在 [0,255] 边缘的微小越界（增强产生的浮点误差）会被裁剪回 [0,255]
    - 其他 dtype（int16 / bool ...）：直接拒绝
    C++：img.convertTo(f, CV_32F, 1.0 / 255.0)"""
    if img.dtype == np.uint8:
        return img.astype(np.float32) * _INV255
    if not np.issubdtype(img.dtype, np.floating):
        raise TypeError("像素 dtype 必须是 uint8 或 float（0~255 量纲），当前 %s" % img.dtype)
    x = img.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("输入图像含有 NaN / Inf")
    lo, hi = float(x.min()), float(x.max())
    if lo < -0.5 or hi > 255.5:
        raise ValueError("float 输入的像素范围应为 0~255，当前 [%.4g, %.4g]；"
                         "如果它已经是 0~1 或已归一化的数据，请不要再次经过 normalize()" % (lo, hi))
    return np.clip(x, 0.0, 255.0) * _INV255


def normalize(img, spec):
    """步骤 4：像素统一成 [0,1]（含范围检查）后做 (x - mean) / std，逐通道，全程 float32，
    mean/std 是 spec 里的 0~1 量纲。灰度图返回 HxWx1。
    全仓唯一的归一化实现：训练数据集、可视化、推理、golden 都走这里。"""
    x = to_unit_range(img)
    if x.ndim == 2:
        x = x[..., None]
    if x.shape[2] != spec.in_channels:
        raise ValueError("图像通道数 %d 与 spec.in_channels=%d 不一致" % (x.shape[2], spec.in_channels))
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(spec.std, dtype=np.float32).reshape(1, 1, -1)
    return (x - mean) / std


def to_blob(x):
    """步骤 5：HxWxC float32 -> 1xCxHxW 连续内存（与 cv2.dnn.blobFromImage 的排布一致）。"""
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


# --------------------------------------------------------------------------
# 组合入口
# --------------------------------------------------------------------------
def _as_model_color(img, spec):
    """检查“已经是模型色彩”的数组：1 通道 -> HxW（HxWx1 会压成 HxW），3 通道 -> HxWx3（RGB）。不做任何颜色转换。"""
    c = spec.in_channels
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if (c == 1 and img.ndim == 2) or (c == 3 and img.ndim == 3 and img.shape[2] == 3):
        return img
    raise ValueError("model_color=True 要求输入已经是模型色彩（%s），当前 shape=%s"
                     % ("灰度 HxW" if c == 1 else "RGB HxWx3", tuple(img.shape)))


def prepare_image(src, spec, model_color=False):
    """步骤 1~3（前半段）：读图/转通道/缩放，返回 uint8 的 RGB HxWx3 或灰度 HxW（模型输入尺寸）。
    src 可以是文件路径，或 OpenCV 的 BGR / 灰度 uint8 数组。
    model_color=True：数组已经是模型色彩（RGB/灰度，如训练数据集 get_full 给的图），跳过步骤 2，避免二次 BGR->RGB。"""
    if isinstance(src, (str, os.PathLike)):
        img = to_model_color(read_image(src, spec.in_channels), spec.in_channels)
    else:
        if src.dtype != np.uint8:
            raise TypeError("推理输入必须是 uint8 图像，当前 dtype=%s" % src.dtype)
        img = _as_model_color(src, spec) if model_color else to_model_color(src, spec.in_channels)
    return resize(img, spec.img_size)


def preprocess(src, spec, model_color=False):
    """完整预处理（唯一对外入口）：
    src（路径，或 BGR/灰度 uint8 数组；model_color=True 时为 RGB/灰度 uint8）
      -> 步骤 1~3 读图/转色/缩放 -> 步骤 4 像素范围检查 + /255 + (x-mean)/std -> 步骤 5 NCHW
    返回 float32 的 1xCxHxW blob。pth / onnxruntime / OpenCV DNN / 训练端可视化都吃它。"""
    return to_blob(normalize(prepare_image(src, spec, model_color), spec))


def load_model_image(path, in_channels):
    """步骤 1+2：给训练端数据集用（RGB 或灰度 uint8）。"""
    return to_model_color(read_image(path, in_channels), in_channels)


# --------------------------------------------------------------------------
# 后处理（C++ 同样需要重写，放在一起便于对照）
# --------------------------------------------------------------------------
def argmax_mask(logits):
    """logits (1,C,h,w) 或 (C,h,w) -> (h,w) uint8 类别图。
    并列最大值取下标最小的那个（np.argmax 的行为）；C++ 手写循环时请用严格大于 `>` 更新最大值。"""
    lg = logits[0] if logits.ndim == 4 else logits
    return lg.argmax(0).astype(np.uint8)


def resize_mask_nearest(mask, orig_hw):
    """把模型输入尺寸上的类别图最近邻放大回原图 (H, W)（不产生新类别）。C++：cv::resize(..., INTER_NEAREST)。"""
    h, w = orig_hw
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


# ==========================================================================
# 以下是自检 / golden 向量生成（直接运行本文件时执行），不属于部署代码
# ==========================================================================
def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("需要 true/false，收到: %r" % v)


def _parse_args(defaults, description=None):
    """与仓库 utils.parse_args 同样的约定（为了本文件不依赖 utils，这里独立实现一份）。"""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k, v in defaults.items():
        if isinstance(v, bool):
            p.add_argument("--" + k, type=_str2bool, default=v)
        elif isinstance(v, (list, tuple)):
            et = type(v[0]) if len(v) else str
            p.add_argument("--" + k, type=lambda s, t=et: [t(x) for x in s.split(",") if x != ""], default=list(v))
        else:
            p.add_argument("--" + k, type=type(v) if v is not None else str, default=v)
    return vars(p.parse_args())


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_spec(cfg):
    """优先从 meta 构造规格；找不到就用 cfg 里的 norm/in_channels/img_size 备用值。返回 (spec, 来源说明)。"""
    path = None
    if cfg["meta"] in ("", "latest"):
        cands = glob.glob(str(_abs(cfg["runs_dir"]) / "**" / "model_meta.json"), recursive=True)
        path = Path(max(cands, key=os.path.getmtime)) if cands else None
    else:
        path = _abs(cfg["meta"])
        if not path.is_file():
            raise FileNotFoundError("meta 文件不存在: %s" % path)
    if path is not None:
        return PreprocessSpec.from_meta(json.loads(path.read_text(encoding="utf-8"))), "meta: %s" % path
    return PreprocessSpec.from_config(cfg), "config 备用值 norm=%s（没找到 model_meta.json）" % cfg["norm"]


def _synthetic(channels, hw, seed=0):
    """合成一张有渐变和噪声的 BGR/灰度 uint8 图，用来覆盖缩小(INTER_AREA)与放大(INTER_LINEAR)两个分支。"""
    h, w = hw
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (xx / max(w - 1, 1) * 160 + yy / max(h - 1, 1) * 80).astype(np.float32)
    bgr = base[..., None] + rng.randn(h, w, 3).astype(np.float32) * 12 + np.array([0, 30, 60], np.float32)
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return bgr if channels == 3 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _opencv_arith_normalize(img_u8, spec):
    """用 convertTo 同款的 /255 + cv2.subtract / cv2.divide + Scalar 做归一化，等价于 C++ 里的写法。"""
    f = img_u8.astype(np.float32) * _INV255                   # C++: convertTo(f, CV_32F, 1.0/255.0)
    pad = lambda v: tuple(float(x) for x in v) + (0.0,) * (4 - len(v))
    f = cv2.subtract(f, pad(spec.mean))
    f = cv2.divide(f, pad(spec.std))
    return f[..., None] if f.ndim == 2 else f


def _check_case(name, img, spec):
    """对一张 BGR/灰度图检查：形状/类型、与 blobFromImage 的排布一致、与 OpenCV 算术写法一致。返回是否通过。"""
    ok = True
    blob = preprocess(img, spec)
    c, (h, w) = spec.in_channels, spec.img_size
    good = blob.shape == (1, c, h, w) and blob.dtype == np.float32 and bool(np.isfinite(blob).all())
    print("  [%s] 输入 %s -> blob %s %s %s" % (name, tuple(img.shape), blob.shape, blob.dtype, "OK" if good else "FAIL"))
    ok &= good

    u8 = prepare_image(img, spec)
    x = normalize(u8, spec)
    try:   # 与 cv2.dnn.blobFromImage 的 HWC->NCHW 排布逐元素一致（C++ 第 5 步就是它）
        ref = cv2.dnn.blobFromImage(x, 1.0, (0, 0), (0, 0, 0, 0), False, False, cv2.CV_32F)
        same = ref.shape == blob.shape and bool(np.array_equal(ref, blob))
        print("        blobFromImage 排布 vs to_blob: %s" % ("逐元素一致" if same else "不一致!"))
        ok &= same
    except Exception as e:  # opencv 没带 dnn 模块时跳过
        print("        [skip] blobFromImage 对比不可用: %s" % e)

    arith = _opencv_arith_normalize(u8, spec)
    diff = float(np.abs(arith - x).max())
    near = diff <= 1e-5
    print("        convertTo(1/255)+subtract/divide 写法 vs numpy: 最大绝对差 %.2e %s" % (diff, "OK" if near else "过大!"))
    ok &= near
    return ok


def _check_range_guard(spec):
    """量纲 / 范围守卫自检：
    1) uint8 的 0 和 255 经 normalize 后必须正好是 (0-mean)/std 与 (1-mean)/std（证明像素确实被缩放到 0~1）
    2) float 输入超出 0~255、含 NaN 必须报错"""
    c = spec.in_channels
    shape = (1, 2) if c == 1 else (1, 2, c)
    u8 = np.zeros(shape, np.uint8)
    u8[0, 1] = 255
    y = normalize(u8, spec)
    m, s = np.asarray(spec.mean, np.float32), np.asarray(spec.std, np.float32)
    ok = bool(np.allclose(y[0, 0], (0.0 - m) / s, atol=1e-6) and np.allclose(y[0, 1], (1.0 - m) / s, atol=1e-6))
    print("  [量纲] uint8 0/255 -> (0-mean)/std 与 (1-mean)/std: %s" % ("OK" if ok else "FAIL"))

    rejected = 0
    for bad in (np.full(shape, 300.0, np.float32), np.full(shape, np.nan, np.float32)):
        try:
            normalize(bad, spec)
        except ValueError:
            rejected += 1
    good = rejected == 2
    print("  [范围] float 越界(300) / NaN 输入被拒绝: %s" % ("OK" if good else "FAIL（守卫没生效）"))
    return ok and good


def _check_against_training(png_path, spec, blob):
    """与训练端旧版 utils.load_image / resize_image / normalize（0~255 量纲的老接口）对比。
    utils.normalize 内部会把 0~255 量纲自动换算成 0~1，所以与新版 normalize() 数值等价，用 1e-5 容差。
    只有 import 失败才 skip；调用报错视为失败（避免量纲守卫之类的问题把这项检查静默变成空跑）。"""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from utils import load_image, normalize as train_normalize, resize_image
    except Exception as e:
        print("  [skip] 找不到训练端 utils（%s: %s），跳过对比" % (type(e).__name__, e))
        return True
    try:
        ref = train_normalize(resize_image(load_image(png_path, spec.in_channels), spec.img_size),
                              spec.mean_255(), spec.std_255())[None]
    except Exception as e:
        print("  [FAIL] 训练端 utils 旧接口调用报错（%s: %s）" % (type(e).__name__, e))
        return False
    diff = float(np.abs(ref - blob).max())
    same = ref.shape == blob.shape and diff <= 1e-5
    print("  与训练端 utils 旧接口对比: 最大绝对差 %.3e -> %s" % (diff, "一致" if same else "不一致，请检查!"))
    return same


def main():
    cfg = _parse_args(DEFAULTS, "预处理自检 + 生成 golden 向量")
    spec, src_txt = _load_spec(cfg)
    c, (h, w) = spec.in_channels, spec.img_size
    print("=" * 78)
    print("预处理规格  来源: %s" % src_txt)
    print("  norm=%s  in_channels=%d  img_size=%dx%d" % (spec.norm, c, h, w))
    print("  mean(0~1)=%s  std(0~1)=%s" % ([round(v, 4) for v in spec.mean], [round(v, 4) for v in spec.std]))
    print("  mean(0~255)=%s  std(0~255)=%s   <- RKNN config 用这组" % (
        [round(v, 3) for v in spec.mean_255()], [round(v, 3) for v in spec.std_255()]))
    print("-" * 78)

    # 主用例：用户给的图（或合成图）-> 存成无损 PNG -> 从 PNG 读回来再预处理，
    # 这样 golden 完全可复现，也排除了 JPEG 解码器差异对对拍的干扰
    out_dir = _abs(cfg["out_dir"])
    if cfg["image"]:
        main_img = read_image(_abs(cfg["image"]), c)
    else:
        main_img = _synthetic(c, (480, 640))
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "golden_input.png"
    ok, buf = cv2.imencode(".png", main_img)
    buf.tofile(str(png_path))
    img = read_image(png_path, c)

    results = [_check_range_guard(spec), _check_case("主用例", img, spec)]
    # 额外覆盖两个缩放分支：比模型输入大（AREA）、比模型输入小（LINEAR）
    results.append(_check_case("放大分支", _synthetic(c, (max(h // 2, 8), max(w // 2, 8)), seed=1), spec))
    results.append(_check_case("缩小分支", _synthetic(c, (h * 2, w * 2), seed=2), spec))

    blob = preprocess(png_path, spec)
    if cfg["check_against_training"]:
        results.append(_check_against_training(png_path, spec, blob))

    if cfg["save_golden"]:
        np.save(out_dir / "golden_resized_u8.npy", prepare_image(img, spec))   # 步骤 1~3 的结果，逐步定位 C++ 差异用
        np.save(out_dir / "golden_blob.npy", blob)                            # 完整预处理结果
        (out_dir / "golden_spec.json").write_text(
            json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print("-" * 78)
        print("golden 已保存到 %s" % out_dir)
        print("  golden_input.png         输入图（无损 PNG，C++ 用 cv::imread 读它）")
        print("  golden_resized_u8.npy    步骤 1~3 的 uint8 结果（RGB 或灰度，HxW[xC]）")
        print("  golden_blob.npy          最终 float32 1xCxHxW blob")
        print("  golden_spec.json         规格与每一步的定义（mean/std 为 0~1 量纲，另附 0~255 版本）")
        print("  C++ 验收：resized 要求逐元素相等；blob 要求最大绝对差 <= 1e-5")
    print("=" * 78)
    print("全部通过" if all(results) else "存在未通过的检查，请看上面的 FAIL / 不一致")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()