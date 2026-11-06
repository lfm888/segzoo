"""推理预处理（pth / onnxruntime / OpenCV DNN 共用），只依赖 numpy + cv2，方便逐行翻译成 C++。

约定：输入是 OpenCV 原生的 BGR uint8（cv2.imread / 摄像头帧）或灰度 uint8；输出是 float32 的 NCHW blob。
归一化 (x-mean)/std 在这里（模型外部）完成，ONNX 里不含归一化。

流水线（每一步在 C++ 里都有同名的 OpenCV 调用，Python 与 C++ 的数值定义完全一致）：

  步骤  Python（本文件）                            C++ (OpenCV)
  ----  ----------------------------------------  ----------------------------------------------------
  1     read_image   cv2.imdecode(IMREAD_COLOR|GRAYSCALE)   cv::imread(path, cv::IMREAD_COLOR|IMREAD_GRAYSCALE)
  2     to_model_color  BGR->RGB / BGR->GRAY               cv::cvtColor(img, img, cv::COLOR_BGR2RGB / COLOR_BGR2GRAY)
  3     resize       任一边要缩小 INTER_AREA，否则 INTER_LINEAR   cv::resize(img, img, cv::Size(W,H), 0, 0, interp)
  4     normalize    uint8->float32, (x-mean)/std（逐通道）  img.convertTo(f, CV_32F);
                                                             cv::subtract(f, cv::Scalar(mean...), f);
                                                             cv::divide(f, cv::Scalar(std...), f);
  5     to_blob      HWC -> 1xCxHxW，float32，连续内存          cv::dnn::blobFromImage(f, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F)

注意：
  * 第 4 步不能用 blobFromImage 的 mean/scalefactor 代替：scalefactor 只能是标量，而 3 通道的 std 逐通道不同。
    所以 C++ 里用 cv::subtract / cv::divide + Scalar 逐通道处理（mean/std 通道顺序 = RGB 顺序，因为第 2 步已经转成 RGB），
    最后的 blobFromImage 只做 HWC->NCHW 的排布变换（swapRB=false，mean 为空，scalefactor=1）。
  * 第 3 步的规则与训练完全一致（utils.resize_image）：只要宽或高有一边需要缩小，两个方向都用 INTER_AREA，
    否则用 INTER_LINEAR；尺寸本来就一样时不做缩放。
  * 后处理见文件末尾：argmax 取“第一个最大值”（与 np.argmax 并列时一致），再最近邻放大回原图。

直接运行本文件 = 自检 + 生成 golden 向量（给 C++ 对拍用）：
  python inference/preprocess.py
  python inference/preprocess.py --image data/xxx/train_dataset/001.png --meta runs/xxx/model_meta.json
"""
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent  # segzoo 仓库根目录（inference/ 的上一级）

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "meta": "latest",                 # model_meta.json / best.pth 同目录导出的元数据；latest = runs 下最新的 model_meta.json
    "runs_dir": "runs",
    "image": "",                      # 自检用的图像；空 = 用程序合成的图
    "out_dir": "inference/golden",    # golden 向量输出目录（相对仓库根目录）
    "save_golden": True,              # 是否保存 golden 向量
    "check_against_training": True,   # 与训练端 utils.load_image/resize_image/normalize 逐元素对比（需要在仓库里运行）
    # 下面四项仅在找不到 meta 时作为备用（没训练过模型也能先跑自检）
    "in_channels": 1,
    "img_size": [288, 512],           # [H, W]
    "mean": [127.5],
    "std": [127.5],
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
    mean: Tuple[float, ...]     # 0~255 单位，长度 == in_channels，通道顺序 = RGB
    std: Tuple[float, ...]

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
        self.in_channels, self.img_size, self.mean, self.std = c, (h, w), mean, std

    @classmethod
    def from_meta(cls, meta):
        """从 train.py 写入 checkpoint 的 meta，或 export_onnx.py 写出的 model_meta.json 构造。
        如果 meta 里有 "preprocess" 子字典就优先用它，否则读顶层的 in_channels/img_size/mean/std。"""
        m = meta.get("preprocess") or meta
        return cls(in_channels=m["in_channels"], img_size=m["img_size"], mean=m["mean"], std=m["std"])

    def to_dict(self):
        return {"in_channels": self.in_channels, "img_size": list(self.img_size),
                "mean": list(self.mean), "std": list(self.std),
                "color_order": "RGB" if self.in_channels == 3 else "GRAY",
                "resize": "INTER_AREA if (src_h > H or src_w > W) else INTER_LINEAR; skip if same size",
                "normalize": "(x - mean) / std in float32, per channel, 0~255 units",
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


def normalize(img, spec):
    """步骤 4：uint8（或增强后的 float）-> float32 的 HxWxC，(x - mean) / std，逐通道，全程 float32。
    灰度图返回 HxWx1。"""
    x = img.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(spec.std, dtype=np.float32).reshape(1, 1, -1)
    return (x - mean) / std


def to_blob(x):
    """步骤 5：HxWxC float32 -> 1xCxHxW 连续内存（与 cv2.dnn.blobFromImage 的排布一致）。"""
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


# --------------------------------------------------------------------------
# 组合入口
# --------------------------------------------------------------------------
def prepare_image(src, spec):
    """步骤 1~3（前半段）：读图/转通道/缩放，返回 uint8 的 RGB HxWx3 或灰度 HxW（模型输入尺寸）。
    src 可以是文件路径，或 OpenCV 的 BGR / 灰度 uint8 数组。"""
    if isinstance(src, (str, os.PathLike)):
        img = read_image(src, spec.in_channels)
    else:
        img = src
        if img.dtype != np.uint8:
            raise TypeError("推理输入必须是 uint8 图像，当前 dtype=%s" % img.dtype)
    return resize(to_model_color(img, spec.in_channels), spec.img_size)


def preprocess(src, spec):
    """完整预处理：src（路径或 BGR/灰度 uint8 数组）-> float32 的 1xCxHxW blob。pth / onnxruntime / OpenCV DNN 都吃它。"""
    return to_blob(normalize(prepare_image(src, spec), spec))


def load_model_image(path, in_channels):
    """步骤 1+2：给训练端数据集用，结果与旧的 utils.load_image(path, in_channels) 完全相同（RGB 或灰度 uint8）。"""
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
    """优先从 meta 构造规格；找不到就用 DEFAULTS 里的备用值。返回 (spec, 来源说明)。"""
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
    spec = PreprocessSpec(cfg["in_channels"], cfg["img_size"], cfg["mean"], cfg["std"])
    return spec, "DEFAULTS 备用值（没找到 model_meta.json）"


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
    """用 cv2.subtract / cv2.divide + Scalar 做归一化，等价于 C++ 里的 cv::subtract / cv::divide（C++ 版会用的写法）。"""
    f = img_u8.astype(np.float32)
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
    print("        cv::subtract/divide 写法 vs numpy: 最大绝对差 %.2e %s" % (diff, "OK" if near else "过大!"))
    ok &= near
    return ok


def _check_against_training(png_path, spec, blob):
    """与训练端 utils.load_image / resize_image / normalize 逐元素对比（要求在 segzoo 仓库里运行）。"""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from utils import load_image, normalize as train_normalize, resize_image
    except Exception as e:
        print("  [skip] 找不到训练端 utils（%s: %s），跳过与训练端对比" % (type(e).__name__, e))
        return True
    ref = train_normalize(resize_image(load_image(png_path, spec.in_channels), spec.img_size),
                          list(spec.mean), list(spec.std))[None]
    diff = float(np.abs(ref - blob).max())
    same = ref.shape == blob.shape and diff == 0.0
    print("  与训练端 utils 的预处理对比: 最大绝对差 %.3e -> %s" % (diff, "逐元素一致" if same else "不一致，请检查!"))
    return same


def main():
    cfg = _parse_args(DEFAULTS, "预处理自检 + 生成 golden 向量")
    spec, src_txt = _load_spec(cfg)
    c, (h, w) = spec.in_channels, spec.img_size
    print("=" * 78)
    print("预处理规格  来源: %s" % src_txt)
    print("  in_channels=%d  img_size=%dx%d  mean=%s  std=%s" % (c, h, w, [round(v, 3) for v in spec.mean],
                                                               [round(v, 3) for v in spec.std]))
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

    results = [_check_case("主用例", img, spec)]
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
        print("  golden_spec.json         规格与每一步的定义")
        print("  C++ 验收：resized 要求逐元素相等；blob 要求最大绝对差 <= 1e-5")
    print("=" * 78)
    print("全部通过" if all(results) else "存在未通过的检查，请看上面的 FAIL / 不一致")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()