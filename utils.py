"""通用工具：命令行默认参数解析、路径解析、图像读写/预处理、可视化。

约定：每个脚本顶部的 DEFAULTS 字典集中放所有参数，直接运行即可；
命令行 --xxx 仍可覆盖（例如 python train.py --arch fpn --epochs 50）。
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from inference import preprocess as pp   # 预处理单一来源（纯 numpy/cv2，无循环依赖）

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
            # nargs="?" 让 "--flag"（等价 true）和 "--flag false" 两种写法都可用
            p.add_argument(flag, type=_str2bool, nargs="?", const=True, default=v)
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


# 预处理实现统一在 inference/preprocess.py（推理端与 C++ 的唯一契约），下面三个函数只是薄封装，
# 保留是为了兼容旧的调用习惯。数值与推理端逐元素一致，不要再在这里写第二份实现。
def load_image(path, in_channels):
    """in_channels=1 读灰度 HxW；否则读 RGB HxWx3。均为 uint8。等价于 inference.preprocess.load_model_image。"""
    return pp.load_model_image(path, in_channels)


def resize_image(img, hw, is_mask=False):
    """图像：走推理端同一套插值规则（缩小用 INTER_AREA，否则 INTER_LINEAR）；
    mask：最近邻（不产生新类别），只在训练/评估端使用，不属于推理流水线。"""
    h, w = hw
    if img.shape[0] == h and img.shape[1] == w:
        return img
    if is_mask:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return pp.resize(img, (h, w))


def normalize(img, mean, std):
    """(x - mean) / std，mean/std 以 0~255 为单位，返回 float32 的 CxHxW。
    等价于 inference.preprocess.normalize(...).transpose(2,0,1)。"""
    x = img.astype(np.float32)
    c = 1 if x.ndim == 2 else x.shape[2]
    spec = pp.PreprocessSpec(c, (1, 1), mean, std)      # normalize 不用 img_size，占位即可
    return np.ascontiguousarray(pp.normalize(x, spec).transpose(2, 0, 1))


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

    col_share = conf.sum(0) / max(total, 1.0)      # 预测成每一类的像素占比
    top = int(np.argmax(col_share)) if num_classes else 0

    return {"dice": dice.tolist(), "iou": iou.tolist(), "prec": prec.tolist(), "rec": rec.tolist(),
            "f1": f1.tolist(),
            "pacc": pacc, "acc_fg": acc_fg, "fg_ratio": float(fg_total / max(total, 1.0)),
            "pred_top_class": top, "pred_top_share": float(col_share[top]),
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


# --------------------------------------------------------------------------
# 进度条 & 人类可读的数字 / 时间（train.py 与 predict.py 共用）
# --------------------------------------------------------------------------
def fmt_secs(s):
    """秒 -> 人类可读（35s / 4m30s / 1h02m）"""
    s = int(round(s))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


def fmt_num(n):
    """大数字 -> 紧凑单位：1_234 -> 1.234 K，105_275_921 -> 105.276 M"""
    n = float(n)
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000 or unit == "T":
            return ("%.3f %s" % (n, unit)).strip() if unit else "%d" % n
        n /= 1000.0


class Progress:
    """轻量进度条（不依赖 tqdm）：终端里 \r 原地刷新，非终端每 20% 打一行。

    info 可以是任意短文本（训练传 "loss 0.123"，预测传文件名）。
    """

    def __init__(self, total, prefix="", enabled=True, info_fmt="%s"):
        self.total, self.prefix, self.i, self.t0 = int(total), prefix, 0, time.time()
        self.live = bool(enabled) and self.total > 0 and sys.stdout.isatty()
        self.enabled = bool(enabled)
        self.info_fmt = info_fmt
        self._w = 0

    def update(self, i, info=""):
        self.i = i
        if not self.enabled or self.total <= 0:
            return
        el = time.time() - self.t0
        tail = ("  " + self.info_fmt % info) if str(info) else ""
        if not self.live:
            step = max(1, self.total // 5)
            if i % step == 0 or i == self.total:
                print("        %s %d/%d (%d%%)%s  用时 %s" % (
                    self.prefix, i, self.total, 100 * i // self.total, tail, fmt_secs(el)))
            return
        frac = i / self.total
        filled = int(24 * frac)
        bar = "#" * filled + "-" * (24 - filled)
        eta = el / max(frac, 1e-9) - el
        line = "  %s [%s] %d/%d %3d%%%s  %s<%s" % (
            self.prefix, bar, i, self.total, int(100 * frac), tail, fmt_secs(el), fmt_secs(eta))
        self._w = max(self._w, len(line))
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def close(self):
        if self.live:
            sys.stdout.write("\r" + " " * (self._w + 1) + "\r")
            sys.stdout.flush()


# --------------------------------------------------------------------------
# 模型复杂度：参数量（按 dtype）、Buffer、MACs / FLOPs、算子占比
# --------------------------------------------------------------------------
def print_model_complexity(model, in_channels, hw, device=None, top_ops=5, verbose=True):
    """打印模型的参数量与计算量，便于跨平台 / 跨模型对比（train.py 和 predict.py 都会调用）。

    - 参数量：model.parameters()，共享权重只算一次；按 dtype 分组
    - Buffer：BN 的 running_mean/var 等，不算参数但占存储，单独列出
    - 计算量：用与 ONNX 导出相同的输入尺寸做一次前向统计（FlopCounterMode），FLOPs = 2 x MACs
    返回 {"params": 参数总量, "trainable": 可训练参数, "bytes": 参数字节数,
          "macs": MACs 或 None, "flops": FLOPs 或 None}
    """
    import torch

    device = device if device is not None else next(model.parameters()).device
    h, w = int(hw[0]), int(hw[1])
    bar = "=" * 60
    print(bar)
    print("模型复杂度统计")
    print(bar)

    # ---------- 1) 参数量，按 dtype 分组 ----------
    by_dtype = {}                                    # dtype -> [个数, 字节数, 可训练个数]
    for p in model.parameters():
        rec = by_dtype.setdefault(p.dtype, [0, 0, 0])
        rec[0] += p.numel()
        rec[1] += p.numel() * p.element_size()
        if p.requires_grad:
            rec[2] += p.numel()
    total_params = sum(r[0] for r in by_dtype.values())
    total_bytes = sum(r[1] for r in by_dtype.values())
    trainable = sum(r[2] for r in by_dtype.values())
    print("参数总量      : %s (%s)" % (format(total_params, ","), fmt_num(total_params)))
    print("可训练参数    : %s (%s)" % (format(trainable, ","), fmt_num(trainable)))
    print("参数占用空间  : %.2f MB" % (total_bytes / 1024 / 1024))
    if by_dtype:
        print("参数数据类型分布:")
        for dt, (n, b, _) in by_dtype.items():
            print("  - %-16s %14s 个 (%s)  %.2f MB" % (str(dt), format(n, ","), fmt_num(n),
                                                      b / 1024 / 1024))

    # ---------- 2) Buffer ----------
    buf_by_dtype = {}
    for b in model.buffers():
        rec = buf_by_dtype.setdefault(b.dtype, [0, 0])
        rec[0] += b.numel()
        rec[1] += b.numel() * b.element_size()
    if buf_by_dtype:
        print("Buffer 数据类型分布（BN 统计量等，不计入参数量）:")
        for dt, (n, b) in buf_by_dtype.items():
            print("  - %-16s %14s 个  %.4f MB" % (str(dt), format(n, ","), b / 1024 / 1024))

    # ---------- 3) 计算量 ----------
    dtype = next(model.parameters()).dtype if total_params else torch.float32
    x = torch.zeros(1, int(in_channels), h, w, dtype=dtype, device=device)
    print("统计输入尺寸  : (1, %d, %d, %d)  dtype=%s" % (int(in_channels), h, w, x.dtype))

    was_training = model.training
    model.eval()
    flops, breakdown, method = None, None, None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad():
            with FlopCounterMode(display=False) as fcm:
                model(x)
        flops = fcm.get_total_flops()
        try:
            breakdown = fcm.get_flop_counts().get("Global", None)
        except Exception:
            breakdown = None
        method = "torch.utils.flop_counter"
    except Exception as e:                            # 老版本 torch / 不支持的算子
        if verbose:
            print("[跳过计算量统计] %s: %s" % (type(e).__name__, e))
    finally:
        if was_training:
            model.train()

    macs = None
    if flops:
        macs = flops / 2
        print("计算量 MACs   : %s (%sMACs)" % (format(int(macs), ","), fmt_num(macs)))
        print("计算量 FLOPs  : %s (%sFLOPs，FLOPs = 2 x MACs)" % (format(int(flops), ","), fmt_num(flops)))
        print("统计方式      : %s" % method)
        if breakdown and verbose:
            top = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[:top_ops]
            print("FLOPs 占比前几的算子:")
            for op, f in top:
                print("  - %-28s %sFLOPs (%.1f%%)" % (str(op), fmt_num(f), f / flops * 100))
    print(bar)
    return {"params": total_params, "trainable": trainable, "bytes": total_bytes,
            "macs": macs, "flops": flops}


# --------------------------------------------------------------------------
# 训练曲线：读 log.csv 画二联图（左 loss+acc，右 per-class Dice）
# --------------------------------------------------------------------------
def plot_training_curves(csv_path, out_path, acc_key="acc_fg", acc_label="Acc(fg)",
                         class_names=None, best_epoch=None, title=None, hide_bg_dice=False, ylim=1.0):
    """把 log.csv 画成一张二联图，返回图片路径或 None：
      左：train/valid 的 loss（左轴）+ 准确率（右轴）
      右：验证集每类 Dice（hide_bg_dice=True 时不画背景类，只看前景）
    两个子图的 y 轴都固定到 [0, ylim]（ylim=1.0 即 0~1），超出范围的点会在图上标注数值。
    matplotlib 不是硬依赖：没装就打印提示并跳过，不影响训练。"""
    import csv as _csv

    try:
        import matplotlib
        matplotlib.use("Agg")
        # 中文类名/数据集名也能正常显示；缺字体时自动退到 DejaVu
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                                  "WenQuanYi Zen Hei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:      # 没装 matplotlib / 无显示环境
        print("[warn] 画训练曲线需要 matplotlib（pip install matplotlib），已跳过: %s" % e)
        return None

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return None

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(name)))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out

    def has(name):
        return name in rows[0]

    ep = [int(float(r["epoch"])) for r in rows]
    dice_cols = [k for k in rows[0] if k.startswith("dice_c")]
    names = list(class_names or []) + ["c%d" % i for i in range(len(dice_cols))]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    fig.suptitle(title or csv_path.stem, fontsize=13, weight="bold")

    # ---------------- 左：loss（左轴） + 准确率（右轴） ----------------
    mk_l = ax_l.axvline(best_epoch, color="#444", ls=":", lw=1.4) if best_epoch else None
    # h1, = ax_l.plot(ep, col("train_loss"), color="#1f77b4", lw=1.6, label="train loss")
    # h2, = ax_l.plot(ep, col("val_loss"), color="#d62728", lw=1.6, label="valid loss")
    C_LOSS, C_ACC = "#1f77b4", "#ff7f0e"      # loss=蓝，准确率=橙；train 实线、valid 虚线
    h1, = ax_l.plot(ep, col("train_loss"), color=C_LOSS, ls="-", lw=1.6, label="train loss")
    h2, = ax_l.plot(ep, col("val_loss"), color=C_LOSS, ls="--", lw=1.6, label="valid loss")

    ax_l.set_title("Loss & Accuracy", fontsize=11)
    ax_l.set_xlabel("epoch")
    # ax_l.set_ylabel("loss")
    ax_l.set_ylabel("loss", color=C_LOSS)
    ax_l.tick_params(axis="y", colors=C_LOSS)
    
    ax_l.grid(alpha=0.3)

    ax_a = ax_l.twinx()
    hs = []
    # if has("train_" + acc_key):
    #     hs.append(ax_a.plot(ep, col("train_" + acc_key), color="#2ca02c", lw=1.5,
    #                         label="train " + acc_label)[0])
    # if has(acc_key):
    #     hs.append(ax_a.plot(ep, col(acc_key), color="#9467bd", lw=1.8,
    #                         label="valid " + acc_label)[0])
    if has("train_" + acc_key):
        hs.append(ax_a.plot(ep, col("train_" + acc_key), color=C_ACC, ls="-", lw=1.6,
                            label="train " + acc_label)[0])
    if has(acc_key):
        hs.append(ax_a.plot(ep, col(acc_key), color=C_ACC, ls="--", lw=1.6,
                            label="valid " + acc_label)[0])
    # ax_a.set_ylabel("accuracy")
    ax_a.set_ylabel("accuracy", color=C_ACC)
    ax_a.tick_params(axis="y", colors=C_ACC)
    handles = [h1, h2] + hs
    if mk_l is not None:
        handles.append(Line2D([], [], color="#444", ls=":", label="best ep %d" % best_epoch))
    # ax_l.legend(handles=handles, fontsize=9, loc="best")
    # 图例放到左图右侧（"accuracy" 纵轴外面），不遮挡曲线；
    # constrained_layout 会自动给它让出空间，右图不会被挤到
    ax_l.legend(handles=handles, fontsize=9, loc="center right")

    # y 轴固定范围（默认 0~1）；loss 超出范围的点在图上标注，避免误解成“acc 到了几十”
    hi = float(ylim) if ylim and float(ylim) > 0 else None
    notes = []
    if hi:
        ax_l.set_ylim(0, hi)
        ax_a.set_ylim(0, 1.0)
        for lbl, k in (("train loss", "train_loss"), ("valid loss", "val_loss")):
            if not has(k):
                continue
            vals = [v for v in col(k) if v == v]
            above = [(v, e) for v, e in zip(col(k), ep) if v == v and v > hi]
            if above:
                v, e = max(above)
                notes.append("%s %.2f@ep%d%s" % (
                    lbl, v, e, "" if len(above) <= 1 else " (+%d more)" % (len(above) - 1)))
    if notes:
        ax_l.text(0.01, 0.015, "clipped above %g (still in log.csv): %s" % (hi, "; ".join(notes)),
                  transform=ax_l.transAxes, fontsize=7.5, color="#666", va="bottom")

    # ---------------- 右：per-class Dice（验证集） ----------------
    if best_epoch:
        ax_r.axvline(best_epoch, color="#444", ls=":", lw=1.4)
    drew = 0
    for i, k in enumerate(dice_cols):
        if hide_bg_dice and i == 0:          # 不画背景类
            continue
        ax_r.plot(ep, col(k), lw=1.6, label=str(names[i])[:14] if i < len(names) else k)
        drew += 1
    ax_r.set_title("Per-class Dice (valid)" + ("  —  foreground only" if hide_bg_dice else ""),
                   fontsize=11)
    ax_r.set_xlabel("epoch")
    ax_r.set_ylabel("Dice")
    ax_r.grid(alpha=0.3)
    if drew:
        ax_r.legend(fontsize=8, ncol=1 if drew <= 6 else 2)

    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    return out_path
