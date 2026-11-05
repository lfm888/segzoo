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


def seg_metrics(conf, num_classes):
    """混淆矩阵（行=真值，列=预测，numpy）-> 各类 dice/iou/P/R/F1 + 前景均值 + 像素准确率。
    train.py（训练/验证）与 predict.py（测试集评估）共用这一套指标。
    验证/测试集里不存在的类，单项指标记 NaN，前景均值只对存在的类求平均（全不存在则记 0）。"""
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diag(conf)
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp

    def ratio(num, den):
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    dice = ratio(2 * tp, 2 * tp + fp + fn)
    iou = ratio(tp, tp + fp + fn)
    prec = ratio(tp, tp + fp)
    rec = ratio(tp, tp + fn)
    f1 = dice

    def fg_mean(x):
        vals = x[1:][~np.isnan(x[1:])] if num_classes > 1 else np.array([])
        return float(vals.mean()) if vals.size else 0.0

    total = float(conf.sum())
    fg_total = float(conf[1:, :].sum()) if num_classes > 1 else 0.0   # 真值是前景的像素数
    pacc = float(tp.sum() / max(total, 1.0))                          # 全部像素准确率
    acc_fg = float(tp[1:].sum() / fg_total) if fg_total > 0 else 0.0   # 只看前景像素的准确率

    return {"dice": dice.tolist(), "iou": iou.tolist(), "prec": prec.tolist(), "rec": rec.tolist(),
            "f1": f1.tolist(),
            "pacc": pacc, "acc_fg": acc_fg, "fg_ratio": float(fg_total / max(total, 1.0)),
            "mdice": fg_mean(dice), "miou": fg_mean(iou),
            "mprec": fg_mean(prec), "mrec": fg_mean(rec), "mf1": fg_mean(f1)}


# --------------------------------------------------------------------------
# 预测结果单图可视化：原图 + 类别半透明填充 + 同类颜色的虚线边界
# --------------------------------------------------------------------------
def palette(n):
    """返回 n 个 BGR 颜色（下标 = 类别索引）；PALETTE 不够时用 HSV 均匀补。"""
    n = max(int(n), 1)
    out = [tuple(int(v) for v in PALETTE[i]) for i in range(min(n, len(PALETTE)))]
    for i in range(len(PALETTE), n):
        hsv = np.uint8([[[int((i * 47) % 180), 220, 255]]])
        b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        out.append((int(b), int(g), int(r)))
    return out


def _point_at(pts, arc, t):
    """在弧长 t 处的插值点"""
    j = int(np.clip(np.searchsorted(arc, t, side="right") - 1, 0, len(pts) - 2))
    seg = arc[j + 1] - arc[j]
    r = 0.0 if seg <= 1e-6 else (t - arc[j]) / seg
    return pts[j] + (pts[j + 1] - pts[j]) * float(np.clip(r, 0, 1))


def _dash_polyline(img, cnt, color, thickness, dash, gap, line_type=cv2.LINE_AA):
    """按弧长把轮廓拆成「实线段 + 空白」画成虚线（OpenCV 没有内置虚线）。"""
    pts = np.asarray(cnt, np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return
    if np.allclose(pts[0], pts[-1]):      # 闭合轮廓：去掉重复终点
        pts = pts[:-1]
    if len(pts) < 3:
        return
    pts = np.vstack([pts, pts[0]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total, period = float(arc[-1]), float(dash + gap)
    t = 0.0
    while t < total:
        a, b = t, min(t + dash, total)
        i0 = int(np.searchsorted(arc, a, side="right") - 1)
        i1 = int(np.searchsorted(arc, b, side="right") - 1)
        seg = [_point_at(pts, arc, a)] + list(pts[max(i0 + 1, 1):i1 + 1]) + [_point_at(pts, arc, b)]
        seg = np.round(np.array(seg, np.float32)).astype(np.int32)
        if len(seg) >= 2:
            cv2.polylines(img, [seg], False, color, thickness, line_type)
        t += period


def _ascii(s, n=16):
    s = "".join(c if 32 <= ord(c) < 127 else "?" for c in str(s))
    return s if len(s) <= n else s[: n - 1] + "~"


def _draw_legend(img, present, colors, names, thickness, line_type=cv2.LINE_AA):
    """左上角图例：虚线小样 + 类别序号/名称（只列当前图里出现的类），字号随图像尺寸缩放。"""
    h, w = img.shape[:2]
    sc = float(np.clip(min(h, w) / 600.0, 0.5, 2.2))
    font, fs, th_txt = cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, max(1, int(round(sc)))
    pad, lh, sw = int(6 * sc), int(18 * sc), int(26 * sc)
    labels = ["%d %s" % (c, _ascii(names[c] if c < len(names) else "class_%d" % c)) for c in present]
    tw = max(cv2.getTextSize(t, font, fs, th_txt)[0][0] for t in labels)
    w_box, h_box = min(tw + sw + 2 * pad, w), min(lh * len(labels) + 2 * pad, h)
    roi = img[0:h_box, 0:w_box].astype(np.float32)
    img[0:h_box, 0:w_box] = (roi * 0.35).astype(np.uint8)   # 半透明黑底，保证文字可读
    for i, (c, txt) in enumerate(zip(present, labels)):
        y = pad + lh * i + lh // 2
        if y + lh // 2 >= h_box:
            break
        cv2.line(img, (pad, y), (pad + int(14 * sc), y), colors[c], max(1, thickness), line_type)
        cv2.putText(img, txt, (pad + sw - int(6 * sc), y + int(5 * sc)), font, fs, colors[c], th_txt, cv2.LINE_AA)
    return img


def render_prediction(img, mask, alpha=0.45, class_names=None, dash=0, gap=0, thickness=None, legend=True,
                      edge_mask=None, draw_edges=True):
    """一张图同时表示原图 / 轮廓 / 分割结果：
      - 每个前景类在原图上用**半透明色填充**（来自 mask = 预测结果）
      - 每类的**边界用同色虚线**画在填充之上（默认也取自 mask；传 edge_mask 就用它，
        典型用法：色块=预测、虚线=真值，一张图直接看预测与 GT 的偏差）
      - draw_edges=False 则不画任何虚线（没有 GT 的场景）
    返回 BGR uint8。img 可以是 load_image 的结果（灰度/RGB）或 resize 后的图，mask 与之同尺寸。
    """
    out = to_bgr(img).astype(np.float32)
    edge = edge_mask if (draw_edges and edge_mask is not None) else (mask if draw_edges else mask)
    n_cls = int(max(int(mask.max()), int(edge.max()) if draw_edges else 0)) + 1
    colors = palette(n_cls)
    fill_cls = [c for c in range(1, n_cls) if (mask == c).any()]
    edge_cls = [c for c in range(1, n_cls) if draw_edges and (edge == c).any()]
    present = sorted(set(fill_cls) | set(edge_cls))
    for c in fill_cls:                                    # 半透明填充（预测）
        m = mask == c
        out[m] = out[m] * (1.0 - alpha) + np.array(colors[c], np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    h, w = out.shape[:2]
    th = int(thickness) if thickness else max(1, int(round(min(h, w) / 600.0)))
    dash = int(dash) if dash and dash > 0 else max(7, 5 * th)     # 虚线：细而能看出节奏
    gap = int(gap) if gap and gap > 0 else max(5, 3 * th)
    line_type = cv2.LINE_8 if th <= 1 else cv2.LINE_AA          # 1 像素线不要抗锯齿，否则虚线看不出间隔
    for c in edge_cls:                                    # 同色虚线边界（默认=预测，可换成真值）
        m = (edge == c).astype(np.uint8)
        cnts = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        for cnt in cnts:
            if len(cnt) >= 6 and cv2.arcLength(cnt, True) >= 8:
                _dash_polyline(out, cnt, colors[c], th, dash, gap, line_type)
    if legend and class_names is not None and present:
        out = _draw_legend(out, present, colors, list(class_names), th, line_type)
    return out
