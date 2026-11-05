"""训练单个模型（架构 x 编码器），保存 best.pth / last.pth / log.csv / results.json / 验证集可视化。

直接运行：python train.py
覆盖参数：python train.py --arch fpn --encoder mobilenet_v2 --epochs 50 --img_size 224,224

本版新增（针对“前几轮验证 loss 异常偏高、验证指标抖动”）：
  1) 逐步线性 warmup + 逐步余弦退火（warmup_epochs / warmup_start_factor / min_lr_factor）
  2) 预训练编码器与解码器分层学习率（encoder_lr_mult）
  3) 预训练权重时自动用 ImageNet 归一化（norm=auto/imagenet/custom），meta 里记录实际 mean/std，
     predict / export_onnx / convert_rknn 读 meta，自动保持一致
  4) 可选早停（early_stop_patience，默认关闭）
"""
import csv
import json
import math
import random
import sys
import time
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import INFO_FILE, build_datasets
from models import benchmark_ms, build_model, count_gflops
from utils import (abspath, imwrite_unicode, normalize, parse_args, plot_training_curves,
                   render_prediction, resize_image, seg_metrics, to_hw)

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    # ---- 数据 ----
    "data_root": "data",              # 数据根目录（相对框架目录，也可写绝对路径如 E:/xxx/data）
    "dataset": "OCT5K",                # 数据集文件夹名：<data_root>/<dataset>/{train_dataset,labels}
    "images_subdir": "train_dataset",
    "labels_subdir": "labels",
    "json_merge_labels": False,       # labelme 里出现多个 label 名时，True=全部并成同一前景类
    "mask_mode": "auto",              # mask 标签编码：auto / index(单通道索引图) / channels(每通道=1类，AIDK 风格)
    "class_names": [],                # 可选类名（含背景也行）；空=自动读 <dataset>/classes.txt，没有则用默认名
    "img_size": [224, 224],           # [H, W]，需为 32 的倍数。AIDK 常用 288,512；labelme 数据用 224,224 或 64,64
    "in_channels": 1,                 # 1=灰度，3=RGB
    "num_classes": 0,                 # 含背景；0=自动（labelme 单类别 -> 2，AIDK 三通道 mask -> 4）
    "norm": "auto",                   # 输入归一化方式：
                                      #   auto     = 用预训练权重(encoder_weights!=none)时用 ImageNet 归一化，否则用下面的 mean/std
                                      #   imagenet = 强制 ImageNet 归一化（3 通道: mean=[123.675,116.28,103.53] std=[58.395,57.12,57.375]；
                                      #              灰度: 取三通道均值作近似）
                                      #   custom   = 用下面的 mean/std
    "mean": [127.5],                  # norm=custom 时使用；(x-mean)/std，0~255 单位；RKNN 转换会读 meta 里实际使用的值
    "std": [127.5],
    "val_ratio": 0.1,                 # 仅用于首次生成 split.txt：验证集占比
    "test_ratio": 0.1,                # 仅用于首次生成 split.txt：测试集占比（train.py 不碰 test，留给 predict.py 评估）
    "split_file": "split.txt",        # 固定划分文件（相对数据集目录）；空字符串=不落盘
    "split_check_ratio": True,        # split.txt 记录的 val_ratio/test_ratio 与当前不一致时自动重新划分（False=永远复用）
    "resplit": False,                 # True = 忽略现有 split.txt 重新生成（默认固定复用，保证可比）
    "split_csv": "auto",              # 首次生成时如有 <dataset>/split/test_train_data.csv 则沿用其 category/fold
                                      # csv 列：image_id(带扩展名),category(train/test/val)，可选 fold 列（fold==0 做 val）
    "seed": 42,
    "cache": True,                    # 预先把全部图缩放后放进内存（大图数据集强烈建议开）
    "label_cache": True,               # 标签扫描结果缓存到 <dataset>/label_info.txt，下次训练不再重复解码 mask
    # ---- 模型 ----
    "arch": "unet",                   # unet / fpn / linknet / pan / unetplusplus / manet / deeplabv3plus ...
    "encoder": "resnet18",            # resnet18 / resnet34 / mobilenet_v2 / tu-xxx(timm 模型) ...
    "encoder_weights": "imagenet",    # imagenet 或 none（下载失败会自动退回 none 并提示）
    # ---- 训练 ----
    "epochs": 100,
    "batch_size": 8,
    "lr": 1e-3,                       # 解码器/分割头学习率（编码器随机初始化时，编码器也用这个）
    "encoder_lr_mult": 0.1,           # 预训练编码器学习率 = lr * 该值（0.1 -> 1e-4）；1.0=不区分。
                                      # 只在 encoder_weights 不是 none 时生效（随机初始化的编码器不该降学习率）
    "warmup_epochs": 2.0,             # 线性 warmup 的轮数（可为小数，按“步”逐步升高）；0=关闭
    "warmup_start_factor": 0.01,      # warmup 起点 = lr * 该值
    "min_lr_factor": 0.01,            # 余弦退火终点 = lr * 该值（与原版 eta_min=lr*0.01 一致）
    "weight_decay": 1e-4,
    "early_stop_patience": 20,         # 早停：验证 mDice 连续多少个 epoch 没有明显提升就停；0=关闭（默认）
                                      # 余弦退火下尾段降学习率常带来最后的提升，建议开启时 patience>=10；
                                      # 适合 sweep 批量对比时省时间，单次精训一般不用
    "early_stop_min_delta": 0.001,    # mDice 提升超过它才算“有提升”
    "early_stop_min_epochs": 10,      # 前若干个 epoch 不触发早停（前期验证指标本来就抖动）
    "loss": "ce_dice",                # ce / dice / ce_dice
    "label_smoothing": 0.0,           # CE 的 label smoothing（建议 0.05）；>0 可避免个别“自信错分”像素把 loss 推到几十
    "amp": True,                      # 混合精度（仅 CUDA）
    "num_workers": 0,                 # 数据已缓存且增强很轻，Windows 下 0 最省事
    "device": "auto",                 # auto / cuda / cpu
    "progress_bar": True,             # 每个 epoch 内的进度条（终端里原地刷新，非终端每 20% 打一行）
    "curve_plot": True,                # 训练结束是否画 loss/准确率/mDice 曲线图 -> curves.png
    "curve_ylim": 1.0,                 # 曲线图 y 轴上限（0 = 自动）；超出的点会在图上标注出来
    "acc_exclude_bg": True,           # True = 打印/汇总里的 Acc 只统计前景像素（不含背景），
                                      #        前景占比很小时才能看出真实分割质量；两种口径都会写进 log.csv / results.json
    "vis_samples": 8,                 # 训练结束后保存多少张验证集可视化（虚线=真值，色块=预测，一张图对比）
    "vis_alpha": 0.45,                # 可视化里分割色块的透明度
    # ---- 数据增强（概率或幅度，设 0 即关闭）----
    "aug_hflip": 0.5,
    "aug_vflip": 0.0,                 # 解剖结构有上下方向（如眼前节）时保持 0
    "aug_affine_p": 0.5,
    "aug_rotate": 10.0,               # 度
    "aug_scale": 0.10,
    "aug_shift": 0.05,                # 占图宽/高的比例
    "aug_color_p": 0.5,               # 外观增强总概率：对比度/亮度/gamma/噪声
    "aug_contrast": 0.3,
    "aug_brightness": 0.15,
    "aug_gamma": 1.4,
    "aug_noise": 0.03,
    "aug_blur_p": 0.1,
    # ---- 输出 ----
    "runs_dir": "runs",               # 结果目录：<runs_dir>/<dataset>/<arch>_<encoder>_<H>x<W>/
}
# ===========================================================================

# ImageNet 统计量（0~255 单位，与 RKNN config 的 mean_values/std_values 同一量纲）
IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def resolve_norm(cfg):
    """决定实际使用的归一化 (mean, std, 说明文字)。
    预训练编码器是在 ImageNet 归一化的输入上训练的，输入分布差太多会让前几轮的 BN 统计量/特征严重错位。"""
    mode = str(cfg.get("norm", "auto")).lower()
    if mode not in ("auto", "imagenet", "custom"):
        raise ValueError("norm 只能是 auto / imagenet / custom，当前: %r" % cfg.get("norm"))
    pretrained = str(cfg["encoder_weights"]).lower() not in ("none", "")
    use_imagenet = mode == "imagenet" or (mode == "auto" and pretrained)
    c = int(cfg["in_channels"])
    if use_imagenet and c == 3:
        return list(IMAGENET_MEAN), list(IMAGENET_STD), "imagenet"
    if use_imagenet and c == 1:
        # smp 会把预训练 conv1 的三通道权重求和用于单通道输入，这里用三通道均值/标准差近似
        return [sum(IMAGENET_MEAN) / 3.0], [sum(IMAGENET_STD) / 3.0], "imagenet(灰度近似)"
    if use_imagenet:
        print("[warn] in_channels=%d 没有对应的 ImageNet 归一化，改用 custom mean/std" % c)
    return [float(v) for v in cfg["mean"]], [float(v) for v in cfg["std"]], "custom"


def make_lr_lambda(total, warm, start, min_f):
    """逐步（按 iteration）的学习率系数：线性 warmup(start->1) 之后接余弦退火(1->min_f)。"""
    def f(step):
        if warm > 0 and step < warm:
            return start + (1.0 - start) * step / warm
        p = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
        return min_f + (1.0 - min_f) * 0.5 * (1.0 + math.cos(math.pi * p))
    return f


def build_optimizer(model, cfg, pretrained):
    """AdamW；预训练编码器用 lr*encoder_lr_mult，解码器/分割头用 lr。返回 (optimizer, 说明文字)。"""
    lr = float(cfg["lr"])
    mult = float(cfg["encoder_lr_mult"]) if pretrained else 1.0
    enc = [p for p in model.encoder.parameters() if p.requires_grad] if hasattr(model, "encoder") else []
    enc_ids = {id(p) for p in enc}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    groups = [{"params": head, "lr": lr}]                      # groups[0] = 解码器/分割头（日志里记它的 lr）
    if enc:
        groups.append({"params": enc, "lr": lr * mult})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    if enc and mult != 1.0:
        txt = "AdamW  解码器 lr=%g / 预训练编码器 lr=%g (×%g)  wd=%g" % (lr, lr * mult, mult, cfg["weight_decay"])
    else:
        txt = "AdamW  lr=%g  wd=%g" % (lr, cfg["weight_decay"])
    return opt, txt


class Criterion(nn.Module):
    def __init__(self, name, label_smoothing=0.0):
        super().__init__()
        self.name = name
        # label_smoothing > 0 可以压住 CE 的“自信错分”尖峰（否则整个验证集均值可能被几像素拉到几十）
        self.ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits, y):
        if self.name == "ce":
            return self.ce(logits, y)
        if self.name == "dice":
            return self.dice(logits, y)
        return self.ce(logits, y) + self.dice(logits, y)


def _fmt_secs(s):
    """秒 -> 人类可读（35s / 4m30s / 1h02m）"""
    s = int(round(s))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


def _short(name, n=12):
    """类名过长时截断，保证一行能放下"""
    name = str(name)
    return name if len(name) <= n else name[: n - 1] + "~"


def conf_metrics(conf, num_classes):
    """torch 混淆矩阵 -> seg_metrics（指标实现统一在 utils，predict.py 复用）。"""
    return seg_metrics(conf.double().cpu().numpy(), num_classes)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, amp):
    """返回 dict：loss + 各类 dice/iou/P/R/F1 + 前景均值 mdice/miou/mprec/mrec/mf1 + 像素准确率 pacc。"""
    model.eval()
    conf = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp):
            logits = model(x)
        loss_sum += criterion(logits.float(), y).item() * x.size(0)
        n += x.size(0)
        pred = logits.argmax(1)
        conf += torch.bincount((y * num_classes + pred).flatten(), minlength=num_classes ** 2)
    m = conf_metrics(conf.view(num_classes, num_classes), num_classes)  # 行=真值 列=预测
    m["loss"] = loss_sum / max(n, 1)
    return m


class Progress:
    """轻量进度条（不依赖 tqdm）：终端里 \\r 原地刷新，非终端每 10% 打一行。"""

    def __init__(self, total, prefix="train", enabled=True):
        self.total, self.prefix, self.i, self.t0 = int(total), prefix, 0, time.time()
        self.live = bool(enabled) and self.total > 0 and sys.stdout.isatty()
        self.enabled = bool(enabled)
        self._w = 0

    def update(self, i, loss):
        self.i = i
        if not self.enabled or self.total <= 0:
            return
        el = time.time() - self.t0
        if not self.live:
            step = max(1, self.total // 5)
            if i % step == 0 or i == self.total:
                print("        %s %d/%d (%d%%)  loss %.4f  用时 %s" % (
                    self.prefix, i, self.total, 100 * i // self.total, loss, _fmt_secs(el)))
            return
        frac = i / self.total
        filled = int(24 * frac)
        bar = "#" * filled + "-" * (24 - filled)
        eta = el / max(frac, 1e-9) - el
        line = "  %s [%s] %d/%d %3d%%  loss %.4f  %s<%s" % (
            self.prefix, bar, i, self.total, int(100 * frac), loss, _fmt_secs(el), _fmt_secs(eta))
        self._w = max(self._w, len(line))
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def close(self):
        if self.live:
            sys.stdout.write("\r" + " " * (self._w + 1) + "\r")
            sys.stdout.flush()


def _clean(v):
    """NaN -> None，便于写 json"""
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


@torch.no_grad()
def save_vis(model, ds, meta, out_dir, device, n, alpha=0.45):
    """验证集可视化，一张图对比 GT 与预测，尺寸与 predict.py 一致（都是原图分辨率）：
      虚线边界 = 真值 mask（同色）；半透明色块 = 预测结果（放大回原图尺寸）；底图 = 原图。"""
    if n <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    names = meta.get("class_names")
    for i in np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int):
        raw, mask = ds.get_full(int(i))          # 原分辨率的图与 GT
        x = torch.from_numpy(normalize(resize_image(raw, ds.hw), meta["mean"], meta["std"]))[None].to(device)
        pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
        # 和 predict.py 一致：预测放大回原图尺寸（最近邻，不产生新类别）
        full = cv2.resize(pred, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
        vis = render_prediction(raw, full, alpha=alpha, class_names=names, edge_mask=mask)
        imwrite_unicode(out_dir / ("val_%03d.png" % i), vis)


W = 78  # 打印行宽


def _kv(lines, k, v):
    """统一的 “键 : 值” 行；按键的显示宽度补空格，中英文混排也能对齐"""
    lines.append("  %s%s" % (k + " " * max(1, 16 - _dwidth(k)), v))


def _dwidth(s):
    """字符串显示宽度（中文字符按 2 列算），用于让分隔线对齐"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))


def _section(title):
    tag = "[%s]" % title
    pad = max(4, W - _dwidth(tag) - 4)
    print("-" * 4 + " " + tag + " " + "-" * pad)


def print_summary(cfg, info, train_ds, val_ds, train_loader, model, C, hw, device, amp, run_dir,
                  norm_txt="", extra_train_lines=()):
    """训练开始前打印：数据集 / 模型(参数量、计算量) / 训练配置 / 输出目录。
    返回 (参数量 M, GFLOPs)，训练结束写 results.json 时直接复用，不再重复统计。
    extra_train_lines: 追加到“训练配置”里的 (键, 值) 行（优化器/学习率策略/早停等）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    shape = (1, int(cfg["in_channels"]), hw[0], hw[1])
    gflops = count_gflops(model, shape, device)  # GFLOPs = 2 * MACs，仅统计卷积/矩阵乘等主要算子
    gflops_txt = "%.3f GFLOPs (%.3f GMACs)" % (gflops, gflops / 2) if gflops is not None else "N/A（需要 torch>=2.1）"

    if info["kind"] == "json":
        kind_txt = "labelme json 多边形"
    elif info.get("mask_mode") == "channels":
        kind_txt = "多通道 mask（每通道=1 类，重叠区后通道覆盖前通道）"
    else:
        kind_txt = "单通道索引图（像素值 %s）" % info["class_values"]
    names = list(info["class_names"][:C]) + ["?"] * max(0, C - len(info["class_names"]))
    classes_txt = "%d（含背景）  " % C + "  ".join(
        "%d=%s" % (i, _short(n, 14)) for i, n in enumerate(names))
    aug_txt = ", ".join("%s=%g" % (k[4:], v) for k, v in cfg.items() if k.startswith("aug_"))
    data_dir = abspath(cfg["data_root"]) / cfg["dataset"]
    n_steps = len(train_loader)

    print("=" * W)
    print("segzoo 训练  |  %s + %s @ %dx%d  |  %s" % (
        cfg["arch"], cfg["encoder"], hw[0], hw[1],
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"))
    print("=" * W)

    _section("1/4 数据集")
    lines = []
    _kv(lines, "数据集", "%s   (%s)" % (cfg["dataset"], data_dir))
    _kv(lines, "样本数", "共 %d 对  ->  训练 %d / 验证 %d / 测试 %d" % (
        len(train_ds) + len(val_ds) + int(info.get("n_test", 0)),
        len(train_ds), len(val_ds), int(info.get("n_test", 0))))
    _kv(lines, "划分方式", "%s" % info.get("split", "-"))
    if info.get("n_test"):
        _kv(lines, "测试集", "%d 张保留不参与训练/验证，模型定型后用 predict.py 评估（--split test）"
            % int(info["n_test"]))
    _kv(lines, "类别数", classes_txt)
    _kv(lines, "标签编码", kind_txt)
    _kv(lines, "标签扫描", "复用缓存 %s" % info["from_cache"] if info.get("from_cache")
        else "本次重新扫描（已缓存到 %s）" % INFO_FILE)
    _kv(lines, "输入", "%d 通道 @ %dx%d" % (cfg["in_channels"], hw[0], hw[1]))
    _kv(lines, "归一化", "%s   mean=%s  std=%s" % (
        norm_txt or "custom", [round(v, 3) for v in cfg["mean"]], [round(v, 3) for v in cfg["std"]]))
    print("\n".join(lines))

    _section("2/4 模型")
    lines = []
    _kv(lines, "结构", "%s + %s" % (cfg["arch"], cfg["encoder"]))
    _kv(lines, "预训练", "%s" % cfg["encoder_weights"])
    _kv(lines, "参数量", "%.3f M / 可训练 %.3f M   (FP32 约 %.1f MB)" % (
        total / 1e6, trainable / 1e6, total * 4 / 1024 ** 2))
    _kv(lines, "计算量", "%s  @ 输入 %s" % (gflops_txt, shape))
    print("\n".join(lines))

    _section("3/4 训练配置")
    lines = []
    _kv(lines, "epochs", "%d  每轮 %d 步  共 %d 步" % (cfg["epochs"], n_steps, cfg["epochs"] * n_steps))
    for k, v in extra_train_lines:
        _kv(lines, k, v)
    _kv(lines, "损失/精度", "loss=%s  AMP=%s  batch=%d  num_workers=%d  cache=%s" % (
        cfg["loss"], amp, cfg["batch_size"], cfg["num_workers"], cfg["cache"]))
    _kv(lines, "数据增强", aug_txt)
    print("\n".join(lines))

    _section("4/4 输出")
    lines = []
    _kv(lines, "目录", "%s" % run_dir)
    _kv(lines, "产物", "best.pth / last.pth / log.csv / results.json / curves.png / vis/val_XXX.png")
    print("\n".join(lines))
    print("=" * W + "\n")
    return total / 1e6, gflops


def print_epoch(ep, n_ep, tm, vm, names, lr, t_train, t_val, elapsed, eta, is_best,
                acc_key="acc_fg", acc_label="Acc(fg)"):
    """每个 epoch 一张小表：train / valid 的 loss、mDice、mIoU、准确率 + 各类 Dice。
    acc_key="acc_fg" 时准确率只算前景像素（不含背景），="pacc" 时算全部像素。"""
    cls = " | ".join("%s %.3f" % (_short(n, 10), d) for n, d in zip(names, vm["dice"]))
    cls_t = " | ".join("%s %.3f" % (_short(n, 10), d) for n, d in zip(names, tm["dice"]))
    print("-" * W)
    print("Epoch %3d/%-3d  lr %.2e  |  训练 %s + 验证 %s = %s  |  已用 %s  剩余 %s" % (
        ep, n_ep, lr, _fmt_secs(t_train), _fmt_secs(t_val), _fmt_secs(t_train + t_val),
        _fmt_secs(elapsed), _fmt_secs(eta)))
    print("  %-6s %-9s %-9s %-9s %-9s %s" % ("phase", "loss", "mDice", "mIoU", acc_label, "各类 Dice"))
    print("  %-6s %-9.4f %-9.4f %-9.4f %-9.4f %s" % (
        "train", tm["loss"], tm["mdice"], tm["miou"], tm[acc_key], cls_t))
    print("  %-6s %-9.4f %-9.4f %-9.4f %-9.4f %s%s" % (
        "valid", vm["loss"], vm["mdice"], vm["miou"], vm[acc_key], cls,
        "   <- best" if is_best else ""))
    print("  valid 前景均值:  Dice %.4f  IoU %.4f  P %.4f  R %.4f  F1 %.4f" % (
        vm["mdice"], vm["miou"], vm["mprec"], vm["mrec"], vm["mf1"]))
    print("  valid 准确率:    全部像素 %.4f  前景像素 %.4f（前景占 %.2f%%）" % (
        vm["pacc"], vm["acc_fg"], 100.0 * vm["fg_ratio"]))
    if vm["pacc"] <= vm["fg_ratio"] * 1.02:      # 不高于“全预测同一类”，基本是塌缩了
        print("  [!] 验证集预测里 %.1f%% 像素都判成 %s 类，像素准确率不高于前景占比 —— "
              "模型可能塌缩；此时 CE 会出现几十的尖峰（-log p），可加 --label_smoothing 0.05 缓解"
              % (100.0 * vm["pred_top_share"], names[vm["pred_top_class"]] if vm["pred_top_class"] < len(names) else vm["pred_top_class"]))
    sys.stdout.flush()


def run(cfg):
    """训练一个模型并返回结果 dict（sweep.py 也会调用它）。"""
    cfg = dict(cfg)                       # 下面会写入实际使用的 mean/std，不改调用方的字典
    # GradScaler 在最开始几步因梯度 inf 跳过 optimizer.step 时，PyTorch 会对逐步 scheduler 报一次无害警告
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step\\(\\).*before.*optimizer.step\\(\\).*")
    t0 = time.time()
    set_seed(cfg["seed"])
    use_cuda = torch.cuda.is_available() and cfg["device"] != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = bool(cfg["amp"]) and use_cuda
    torch.backends.cudnn.benchmark = True

    hw = to_hw(cfg["img_size"])
    if hw[0] % 32 or hw[1] % 32:
        raise ValueError("img_size 必须是 32 的倍数，当前 %s" % (hw,))

    # 归一化必须在建数据集之前决定（数据集用 cfg["mean"]/cfg["std"]），并写进 meta 供 predict/export/rknn 使用
    cfg["mean"], cfg["std"], norm_txt = resolve_norm(cfg)

    train_ds, val_ds, info = build_datasets(cfg)
    C = cfg["num_classes"] or info["num_classes"]
    name = "%s_%s_%dx%d" % (cfg["arch"], cfg["encoder"], hw[0], hw[1])
    run_dir = abspath(cfg["runs_dir"]) / cfg["dataset"] / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print("[run] 构建模型 %s + %s (in_channels=%d, classes=%d, encoder_weights=%s) ..." % (
        cfg["arch"], cfg["encoder"], int(cfg["in_channels"]), C, cfg["encoder_weights"]))

    ew = None if str(cfg["encoder_weights"]).lower() in ("none", "") else cfg["encoder_weights"]
    model = build_model(cfg["arch"], cfg["encoder"], cfg["in_channels"], C, ew).to(device)

    meta = {  # 只放基础类型，保证 torch.load 在新版本下也安全
        "arch": cfg["arch"], "encoder": cfg["encoder"], "in_channels": int(cfg["in_channels"]),
        "num_classes": int(C), "img_size": [hw[0], hw[1]],
        "mean": [float(v) for v in cfg["mean"]], "std": [float(v) for v in cfg["std"]], "norm": norm_txt,
        "dataset": cfg["dataset"], "data_root": str(cfg["data_root"]), "images_subdir": cfg["images_subdir"],
        "labels_subdir": cfg["labels_subdir"], "json_merge_labels": bool(cfg["json_merge_labels"]),
        "kind": info["kind"], "mask_mode": info.get("mask_mode"),
        "class_values": info.get("class_values"), "label2idx": info.get("label2idx"),
        "class_names": list(info["class_names"][:int(C)]),
    }

    bs = int(cfg["batch_size"])
    nw = int(cfg["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=use_cuda,
                              drop_last=len(train_ds) > bs, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=use_cuda,
                            persistent_workers=nw > 0)

    criterion = Criterion(cfg["loss"], cfg.get("label_smoothing", 0.0))

    # 优化器（预训练编码器降学习率）+ 逐步 warmup/余弦调度
    opt, opt_txt = build_optimizer(model, cfg, pretrained=ew is not None)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = int(cfg["epochs"]) * steps_per_epoch
    warm_steps = int(round(float(cfg["warmup_epochs"]) * steps_per_epoch))
    warm_steps = min(max(warm_steps, 0), total_steps // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(total_steps, warm_steps, float(cfg["warmup_start_factor"]), float(cfg["min_lr_factor"])))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    es_pat = int(cfg["early_stop_patience"])
    es_delta = float(cfg["early_stop_min_delta"])
    es_min_ep = int(cfg["early_stop_min_epochs"])
    extra_train_lines = [
        ("优化器", opt_txt),
        ("学习率策略", ("warmup %d 步（%.1f 轮，起点 ×%g）→ 余弦退火（逐步）至 ×%g" % (
            warm_steps, warm_steps / steps_per_epoch, cfg["warmup_start_factor"], cfg["min_lr_factor"]))
         if warm_steps > 0 else ("无 warmup → 余弦退火（逐步）至 ×%g" % cfg["min_lr_factor"])),
        ("早停", ("验证 mDice 连续 %d 轮提升 < %g 则停（前 %d 轮不触发）" % (es_pat, es_delta, es_min_ep))
         if es_pat > 0 else "关闭（early_stop_patience=0）"),
    ]

    # 训练前打印数据集、模型参数量/计算量、训练配置
    params_m, gflops = print_summary(cfg, info, train_ds, val_ds, train_loader, model, C, hw, device, amp, run_dir,
                                     norm_txt=norm_txt, extra_train_lines=extra_train_lines)

    best, best_epoch, best_metrics = -1.0, 0, None
    es_best, es_bad, stopped_at, epochs_run = -1.0, 0, None, 0
    names = list(info["class_names"][:int(C)]) + ["class_%d" % i for i in range(len(info["class_names"]), int(C))]
    t_train_total = 0.0
    use_fg_acc = bool(cfg.get("acc_exclude_bg", True))
    acc_key, acc_label = ("acc_fg", "Acc(fg)") if use_fg_acc else ("pacc", "Acc")
    with open(run_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_mdice", "train_miou",
                    "train_pacc", "train_acc_fg", "mdice", "miou", "pacc", "acc_fg", "lr"]
                   + ["dice_c%d" % c for c in range(int(C))])
        for ep in range(1, cfg["epochs"] + 1):
            model.train()
            tl, tn = 0.0, 0
            conf_t = torch.zeros(int(C) * int(C), dtype=torch.long, device=device)
            prog = Progress(len(train_loader), "train", enabled=bool(cfg.get("progress_bar", True)))
            t_tr = time.time()
            for i, (x, y) in enumerate(train_loader, 1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=amp):
                    logits = model(x)
                loss = criterion(logits.float(), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()                      # 逐步更新学习率（warmup + 余弦）
                tl += loss.item() * x.size(0)
                tn += x.size(0)
                with torch.no_grad():  # 训练集指标顺便统计，便于对比是否过拟合
                    conf_t += torch.bincount(
                        (y * int(C) + logits.argmax(1)).flatten(), minlength=int(C) ** 2)
                prog.update(i, tl / max(tn, 1))
            prog.close()
            t_train = time.time() - t_tr
            tm = conf_metrics(conf_t.view(int(C), int(C)), int(C))
            tm["loss"] = tl / max(tn, 1)

            lr_now = opt.param_groups[0]["lr"]    # 本 epoch 结束时解码器/分割头的学习率

            t_va = time.time()
            m = evaluate(model, val_loader, criterion, device, int(C), amp)
            t_val = time.time() - t_va

            is_best = m["mdice"] > best
            if is_best:
                best, best_epoch, best_metrics = m["mdice"], ep, m
                torch.save({"model": model.state_dict(), "meta": meta}, run_dir / "best.pth")

            w.writerow([ep, tm["loss"], m["loss"], tm["mdice"], tm["miou"],
                        tm["pacc"], tm["acc_fg"], m["mdice"], m["miou"], m["pacc"], m["acc_fg"],
                        lr_now] + m["dice"])
            f.flush()

            t_train_total += t_train + t_val
            eta = t_train_total / ep * (cfg["epochs"] - ep)
            print_epoch(ep, cfg["epochs"], tm, m, names, lr_now, t_train, t_val,
                        time.time() - t0, eta, is_best, acc_key, acc_label)
            epochs_run = ep

            # ---- 早停（默认关闭）：用“提升超过 min_delta”的口径，与 best 的“严格更大即保存”分开 ----
            if m["mdice"] > es_best + es_delta:
                es_best, es_bad = m["mdice"], 0
            else:
                es_bad += 1
            if es_pat > 0 and ep >= es_min_ep and es_bad >= es_pat:
                stopped_at = ep
                print("[早停] 验证 mDice 已连续 %d 轮提升 < %g（最佳 %.4f @ epoch %d），在第 %d 轮停止训练" % (
                    es_bad, es_delta, best, best_epoch, ep))
                break
    torch.save({"model": model.state_dict(), "meta": meta}, run_dir / "last.pth")

    # 用最佳权重做可视化与效率统计
    ck = torch.load(str(run_dir / "best.pth"), map_location="cpu")
    model.load_state_dict(ck["model"])
    save_vis(model, val_ds, meta, run_dir / "vis", device, int(cfg["vis_samples"]),
             alpha=float(cfg.get("vis_alpha", 0.45)))
    shape = (1, int(cfg["in_channels"]), hw[0], hw[1])
    results = {
        "name": name, "arch": cfg["arch"], "encoder": cfg["encoder"], "img_size": [hw[0], hw[1]],
        "num_classes": C, "epochs": cfg["epochs"], "epochs_run": epochs_run,
        "early_stopped": stopped_at is not None, "norm": norm_txt, "best_epoch": best_epoch,
        "mdice": _clean(best_metrics["mdice"]), "miou": _clean(best_metrics["miou"]),
        "mprec": _clean(best_metrics["mprec"]), "mrec": _clean(best_metrics["mrec"]),
        "mf1": _clean(best_metrics["mf1"]), "pacc": _clean(best_metrics["pacc"]),
        "acc_fg": _clean(best_metrics["acc_fg"]), "fg_ratio": _clean(best_metrics["fg_ratio"]),
        "dice": _clean(best_metrics["dice"]), "iou": _clean(best_metrics["iou"]),
        "params_m": round(params_m, 3),
        "gflops": gflops,
        "latency_ms": round(benchmark_ms(model, shape, device), 3),
        "latency_device": device.type, "train_minutes": round((time.time() - t0) / 60, 2),
        "run_dir": str(run_dir),
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # loss / mDice / mIoU / 准确率 / 各类 Dice 曲线图
    curve_path = None
    if cfg.get("curve_plot", True):
        curve_path = plot_training_curves(
            run_dir / "log.csv", run_dir / "curves.png", acc_key=acc_key, acc_label=acc_label,
            class_names=names, best_epoch=best_epoch, hide_bg_dice=use_fg_acc,
            ylim=float(cfg.get("curve_ylim", 1.0)),
            title="%s + %s @ %dx%d" % (cfg["arch"], cfg["encoder"], hw[0], hw[1]) + " — " + cfg["dataset"])
        if curve_path:
            results["curves"] = str(curve_path)
            with open(run_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- 训练结束汇总 ----
    best_cls = " | ".join("%s %.3f" % (_short(n, 10), d) for n, d in zip(names, best_metrics["dice"]))
    print("=" * W)
    print("训练完成  |  %s + %s @ %dx%d  |  总用时 %s" % (
        cfg["arch"], cfg["encoder"], hw[0], hw[1], _fmt_secs(time.time() - t0)))
    print("-" * W)
    lines = []
    _kv(lines, "最佳 epoch", "%d / %d%s" % (best_epoch, cfg["epochs"],
                                          "（早停于第 %d 轮）" % stopped_at if stopped_at else ""))
    _kv(lines, "验证指标", "mDice(fg) %.4f  mIoU %.4f  P %.4f  R %.4f  F1 %.4f" % (
        best_metrics["mdice"], best_metrics["miou"], best_metrics["mprec"], best_metrics["mrec"],
        best_metrics["mf1"]))
    _kv(lines, "准确率", "前景像素(Acc fg) %.4f   全部像素 %.4f   （前景占 %.2f%%）" % (
        best_metrics["acc_fg"], best_metrics["pacc"], 100.0 * best_metrics["fg_ratio"]))
    _kv(lines, "各类 Dice", best_cls)
    _kv(lines, "效率", "%.3f M 参数  |  %.3f GFLOPs  |  %.2f ms/张 (%s)" % (
        results["params_m"], results["gflops"], results["latency_ms"], device.type))
    _kv(lines, "输出", "%s" % run_dir)
    _kv(lines, "产物", "best.pth  last.pth  log.csv  results.json  curves.png  vis/val_XXX.png（虚线=GT, 色块=预测）")
    print("\n".join(lines))
    print("=" * W)
    return results


if __name__ == "__main__":  # Windows 下启用多进程 DataLoader 必须有这个保护
    run(parse_args(DEFAULTS, "训练单个分割模型"))