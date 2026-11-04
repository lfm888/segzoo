"""通用工具：命令行默认参数解析、路径解析、图像读写/预处理、可视化。

约定：每个脚本顶部的 DEFAULTS 字典集中放所有参数，直接运行即可；
命令行 --xxx 仍可覆盖（例如 python train.py --arch fpn --epochs 50）。
"""
import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent  # 本框架所在目录


def abspath(p):
    """相对路径一律相对于框架目录，这样在任何工作目录下运行都一致。"""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# --------------------------------------------------------------------------
# 命令行：根据 DEFAULTS 自动生成 --key 选项（类型取自默认值的类型）
# --------------------------------------------------------------------------
def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("需要 true/false，收到: %r" % v)


def parse_args(defaults, description=None):
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for k, v in defaults.items():
        flag = "--" + k
        if isinstance(v, bool):  # 必须先于 int 判断（bool 是 int 的子类）
            p.add_argument(flag, type=_str2bool, default=v)
        elif isinstance(v, (list, tuple)):
            et = type(v[0]) if len(v) else str
            # 列表参数用逗号分隔：--img_size 288,512  --archs unet,fpn
            p.add_argument(flag, type=lambda s, t=et: [t(x) for x in s.split(",") if x != ""], default=list(v))
        elif v is None:
            p.add_argument(flag, type=str, default=None)
        else:
            p.add_argument(flag, type=type(v), default=v)
    return vars(p.parse_args())


def to_hw(size):
    """int / [s] / [h, w] -> (h, w)"""
    if isinstance(size, (int, float)):
        return int(size), int(size)
    size = list(size)
    if len(size) == 1:
        return int(size[0]), int(size[0])
    return int(size[0]), int(size[1])


def latest_file(runs_dir, filename):
    """在 runs_dir 下递归找最新修改的 filename（如 best.pth / model.onnx）。"""
    cands = glob.glob(str(abspath(runs_dir) / "**" / filename), recursive=True)
    if not cands:
        raise FileNotFoundError("在 %s 下找不到 %s，请先训练/导出" % (abspath(runs_dir), filename))
    return Path(max(cands, key=os.path.getmtime))


def resolve_path(value, runs_dir, filename):
    """value 为空或 'latest' 时自动取最新；否则当作路径。"""
    if value in (None, "", "latest"):
        return latest_file(runs_dir, filename)
    return abspath(value)


# --------------------------------------------------------------------------
# 图像读写（用 fromfile/imdecode，避免 Windows 下 cv2 不能读中文路径）
# --------------------------------------------------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def imread_unicode(path, flag=cv2.IMREAD_UNCHANGED):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flag)
    if img is None:
        raise IOError("读取图像失败: %s" % path)
    return img


def imwrite_unicode(path, img):
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("写图像失败: %s" % path)
    buf.tofile(str(path))


def load_image(path, in_channels):
    """in_channels=1 读灰度 HxW；否则读 RGB HxWx3。均为 uint8。"""
    if in_channels == 1:
        return imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(imread_unicode(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def resize_image(img, hw, is_mask=False):
    h, w = hw
    if img.shape[0] == h and img.shape[1] == w:
        return img
    if is_mask:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    shrink = img.shape[0] > h or img.shape[1] > w
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def normalize(img, mean, std):
    """(x - mean) / std，mean/std 以 0~255 为单位（与 RKNN config 的 mean/std 一致）。
    返回 float32 的 CxHxW。"""
    x = img.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    c = x.shape[2]
    m = np.array(mean if len(mean) == c else [mean[0]] * c, dtype=np.float32)
    s = np.array(std if len(std) == c else [std[0]] * c, dtype=np.float32)
    return np.ascontiguousarray(((x - m) / s).transpose(2, 0, 1))


# --------------------------------------------------------------------------
# 可视化
# --------------------------------------------------------------------------
# BGR 调色板，下标 = 类别索引（0 为背景）
PALETTE = np.array(
    [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 128, 255)],
    dtype=np.uint8,
)


def colorize(mask):
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]


def to_bgr(img):
    """load_image 的结果（灰度 / RGB）-> BGR uint8，用于保存可视化。"""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def overlay(img, mask, alpha=0.5):
    base = to_bgr(img).astype(np.float32)
    col = colorize(mask).astype(np.float32)
    fg = (mask > 0)[..., None]
    out = np.where(fg, base * (1 - alpha) + col * alpha, base)
    return out.astype(np.uint8)
