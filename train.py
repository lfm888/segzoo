"""训练单个模型（架构 x 编码器），保存 best.pth / last.pth / log.csv / results.json / 验证集可视化。

直接运行：python train.py
覆盖参数：python train.py --arch fpn --encoder mobilenet_v2 --epochs 50 --img_size 224,224
"""
import csv
import json
import random
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import INFO_FILE, build_datasets
from models import benchmark_ms, build_model, count_gflops
from utils import (abspath, imwrite_unicode, normalize, parse_args, render_prediction, resize_image,
                   to_hw)

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
    "mean": [127.5],                  # 归一化 (x-mean)/std，0~255 单位；RKNN 转换会读同一组值
    "std": [127.5],
    "val_ratio": 0.2,
    "split_csv": "auto",              # 划分文件：auto=找 <dataset>/split/test_train_data.csv；none=纯随机；或写具体路径
                                      # csv 列：image_id(带扩展名),category(train/test/val)，可选 fold 列（fold==0 做验证集）
    "seed": 42,
    "cache": True,                    # 预先把全部图缩放后放进内存（大图数据集强烈建议开）
    "label_cache": True,               # 标签扫描结果缓存到 <dataset>/label_info.txt，下次训练不再重复解码 mask
    # ---- 模型 ----
    "arch": "unet",                   # unet / fpn / linknet / pan / unetplusplus / manet / deeplabv3plus ...
    "encoder": "resnet18",            # resnet18 / resnet34 / mobilenet_v2 / tu-xxx(timm 模型) ...
    "encoder_weights": "imagenet",    # imagenet 或 none（下载失败会自动退回 none 并提示）
    # ---- 训练 ----
    "epochs": 3,
    "batch_size": 8,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "loss": "dice",                # ce / dice / ce_dice
    "amp": True,                      # 混合精度（仅 CUDA）
    "num_workers": 0,                 # 数据已缓存且增强很轻，Windows 下 0 最省事
    "device": "auto",                 # auto / cuda / cpu
    "progress_bar": True,             # 每个 epoch 内的进度条（终端里原地刷新，非终端每 20% 打一行）
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


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class Criterion(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.ce = nn.CrossEntropyLoss()
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
    """混淆矩阵（行=真值，列=预测）-> 各类 dice/iou/P/R/F1 + 前景均值（不含背景，不存在的类跳过）。"""
    conf = conf.double().cpu()
    tp = conf.diag()
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    nan = torch.tensor(float("nan"), dtype=torch.double)
    dice = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1), nan)
    iou = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn).clamp(min=1), nan)
    prec = torch.where(tp + fp > 0, tp / (tp + fp).clamp(min=1), nan)
    rec = torch.where(tp + fn > 0, tp / (tp + fn).clamp(min=1), nan)

    def fg_mean(x):
        v = float(torch.nanmean(x[1:])) if num_classes > 1 else float("nan")
        return 0.0 if v != v else v   # 前景类都不存在时按 0 处理，避免 best 判断失效

    denom = (2 * tp + fp + fn).clamp(min=1)
    return {"dice": dice.tolist(), "iou": iou.tolist(), "prec": prec.tolist(), "rec": rec.tolist(),
            "f1": (2 * tp / denom).tolist(),
            "pacc": float(tp.sum() / conf.sum().clamp(min=1)),
            "mdice": fg_mean(dice), "miou": fg_mean(iou),
            "mprec": fg_mean(prec), "mrec": fg_mean(rec), "mf1": fg_mean(2 * tp / denom)}


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


def print_summary(cfg, info, train_ds, val_ds, train_loader, model, C, hw, device, amp, run_dir):
    """训练开始前打印：数据集 / 模型(参数量、计算量) / 训练配置 / 输出目录。
    返回 (参数量 M, GFLOPs)，训练结束写 results.json 时直接复用，不再重复统计。"""
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
    _kv(lines, "样本数", "共 %d 对  ->  训练 %d / 验证 %d   (val_ratio=%g, seed=%d)" % (
        len(train_ds) + len(val_ds), len(train_ds), len(val_ds), cfg["val_ratio"], cfg["seed"]))
    _kv(lines, "划分方式", "%s" % info.get("split", "random"))
    _kv(lines, "类别数", classes_txt)
    _kv(lines, "标签编码", kind_txt)
    _kv(lines, "标签扫描", "复用缓存 %s" % info["from_cache"] if info.get("from_cache")
        else "本次重新扫描（已缓存到 %s）" % INFO_FILE)
    _kv(lines, "输入", "%d 通道 @ %dx%d   mean=%s  std=%s" % (
        cfg["in_channels"], hw[0], hw[1], cfg["mean"], cfg["std"]))
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
    _kv(lines, "优化器", "AdamW  lr=%g  wd=%g  余弦退火" % (cfg["lr"], cfg["weight_decay"]))
    _kv(lines, "损失/精度", "loss=%s  AMP=%s  batch=%d  num_workers=%d  cache=%s" % (
        cfg["loss"], amp, cfg["batch_size"], cfg["num_workers"], cfg["cache"]))
    _kv(lines, "数据增强", aug_txt)
    print("\n".join(lines))

    _section("4/4 输出")
    lines = []
    _kv(lines, "目录", "%s" % run_dir)
    _kv(lines, "产物", "best.pth / last.pth / log.csv / results.json / vis/val_XXX.png")
    print("\n".join(lines))
    print("=" * W + "\n")
    return total / 1e6, gflops


def print_epoch(ep, n_ep, tm, vm, names, lr, t_train, t_val, elapsed, eta, is_best):
    """每个 epoch 一张小表：train / valid 的 loss、mDice、mIoU、准确率 + 各类 Dice。"""
    cls = " | ".join("%s %.3f" % (_short(n, 10), d) for n, d in zip(names, vm["dice"]))
    cls_t = " | ".join("%s %.3f" % (_short(n, 10), d) for n, d in zip(names, tm["dice"]))
    print("-" * W)
    print("Epoch %3d/%-3d  lr %.2e  |  训练 %s + 验证 %s = %s  |  已用 %s  剩余 %s" % (
        ep, n_ep, lr, _fmt_secs(t_train), _fmt_secs(t_val), _fmt_secs(t_train + t_val),
        _fmt_secs(elapsed), _fmt_secs(eta)))
    print("  %-6s %-9s %-9s %-9s %-9s %s" % ("phase", "loss", "mDice", "mIoU", "Acc", "各类 Dice"))
    print("  %-6s %-9.4f %-9.4f %-9.4f %-9.4f %s" % ("train", tm["loss"], tm["mdice"], tm["miou"], tm["pacc"], cls_t))
    print("  %-6s %-9.4f %-9.4f %-9.4f %-9.4f %s%s" % (
        "valid", vm["loss"], vm["mdice"], vm["miou"], vm["pacc"], cls,
        "   <- best" if is_best else ""))
    print("  valid 前景均值:  Dice %.4f  IoU %.4f  P %.4f  R %.4f  F1 %.4f" % (
        vm["mdice"], vm["miou"], vm["mprec"], vm["mrec"], vm["mf1"]))
    sys.stdout.flush()


def run(cfg):
    """训练一个模型并返回结果 dict（sweep.py 也会调用它）。"""
    t0 = time.time()
    set_seed(cfg["seed"])
    use_cuda = torch.cuda.is_available() and cfg["device"] != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = bool(cfg["amp"]) and use_cuda
    torch.backends.cudnn.benchmark = True

    hw = to_hw(cfg["img_size"])
    if hw[0] % 32 or hw[1] % 32:
        raise ValueError("img_size 必须是 32 的倍数，当前 %s" % (hw,))

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
        "mean": [float(v) for v in cfg["mean"]], "std": [float(v) for v in cfg["std"]],
        "dataset": cfg["dataset"], "data_root": str(cfg["data_root"]), "images_subdir": cfg["images_subdir"],
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

    criterion = Criterion(cfg["loss"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 训练前打印数据集、模型参数量/计算量、训练配置
    params_m, gflops = print_summary(cfg, info, train_ds, val_ds, train_loader, model, C, hw, device, amp, run_dir)

    best, best_epoch, best_metrics = -1.0, 0, None
    names = list(info["class_names"][:int(C)]) + ["class_%d" % i for i in range(len(info["class_names"]), int(C))]
    t_train_total = 0.0
    with open(run_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_mdice", "train_miou", "mdice", "miou", "pacc", "lr"]
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

            lr_now = opt.param_groups[0]["lr"]
            sched.step()

            t_va = time.time()
            m = evaluate(model, val_loader, criterion, device, int(C), amp)
            t_val = time.time() - t_va

            is_best = m["mdice"] > best
            if is_best:
                best, best_epoch, best_metrics = m["mdice"], ep, m
                torch.save({"model": model.state_dict(), "meta": meta}, run_dir / "best.pth")

            w.writerow([ep, tm["loss"], m["loss"], tm["mdice"], tm["miou"], m["mdice"], m["miou"], m["pacc"], lr_now]
                       + m["dice"])
            f.flush()

            t_train_total += t_train + t_val
            eta = t_train_total / ep * (cfg["epochs"] - ep)
            print_epoch(ep, cfg["epochs"], tm, m, names, lr_now, t_train, t_val,
                        time.time() - t0, eta, is_best)
    torch.save({"model": model.state_dict(), "meta": meta}, run_dir / "last.pth")

    # 用最佳权重做可视化与效率统计
    ck = torch.load(str(run_dir / "best.pth"), map_location="cpu")
    model.load_state_dict(ck["model"])
    save_vis(model, val_ds, meta, run_dir / "vis", device, int(cfg["vis_samples"]),
             alpha=float(cfg.get("vis_alpha", 0.45)))
    shape = (1, int(cfg["in_channels"]), hw[0], hw[1])
    results = {
        "name": name, "arch": cfg["arch"], "encoder": cfg["encoder"], "img_size": [hw[0], hw[1]],
        "num_classes": C, "epochs": cfg["epochs"], "best_epoch": best_epoch,
        "mdice": _clean(best_metrics["mdice"]), "miou": _clean(best_metrics["miou"]),
        "mprec": _clean(best_metrics["mprec"]), "mrec": _clean(best_metrics["mrec"]),
        "mf1": _clean(best_metrics["mf1"]), "pacc": _clean(best_metrics["pacc"]),
        "dice": _clean(best_metrics["dice"]), "iou": _clean(best_metrics["iou"]),
        "params_m": round(params_m, 3),
        "gflops": gflops,
        "latency_ms": round(benchmark_ms(model, shape, device), 3),
        "latency_device": device.type, "train_minutes": round((time.time() - t0) / 60, 2),
        "run_dir": str(run_dir),
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- 训练结束汇总 ----
    best_cls = " | ".join("%s %.3f" % (_short(n, 10), d) for n, d in zip(names, best_metrics["dice"]))
    print("=" * W)
    print("训练完成  |  %s + %s @ %dx%d  |  总用时 %s" % (
        cfg["arch"], cfg["encoder"], hw[0], hw[1], _fmt_secs(time.time() - t0)))
    print("-" * W)
    lines = []
    _kv(lines, "最佳 epoch", "%d / %d" % (best_epoch, cfg["epochs"]))
    _kv(lines, "验证指标", "mDice(fg) %.4f  mIoU %.4f  P %.4f  R %.4f  F1 %.4f  Acc %.4f" % (
        best_metrics["mdice"], best_metrics["miou"], best_metrics["mprec"], best_metrics["mrec"],
        best_metrics["mf1"], best_metrics["pacc"]))
    _kv(lines, "各类 Dice", best_cls)
    _kv(lines, "效率", "%.3f M 参数  |  %.3f GFLOPs  |  %.2f ms/张 (%s)" % (
        results["params_m"], results["gflops"], results["latency_ms"], device.type))
    _kv(lines, "输出", "%s" % run_dir)
    _kv(lines, "产物", "best.pth  last.pth  log.csv  results.json  vis/val_XXX.png（虚线=GT, 色块=预测）")
    print("\n".join(lines))
    print("=" * W)
    return results


if __name__ == "__main__":  # Windows 下启用多进程 DataLoader 必须有这个保护
    run(parse_args(DEFAULTS, "训练单个分割模型"))