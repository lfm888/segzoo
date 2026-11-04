"""训练单个模型（架构 x 编码器），保存 best.pth / last.pth / log.csv / results.json / 验证集可视化。

直接运行：python train.py
覆盖参数：python train.py --arch fpn --encoder mobilenet_v2 --epochs 50 --img_size 224,224
"""
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import build_datasets
from models import benchmark_ms, build_model, count_gflops, count_params
from utils import abspath, imwrite_unicode, normalize, overlay, parse_args, to_bgr, to_hw

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    # ---- 数据 ----
    "data_root": "data",              # 数据根目录（相对框架目录，也可写绝对路径如 E:/xxx/data）
    "dataset": "AIDK",                # 数据集文件夹名：<data_root>/<dataset>/{train_dataset,labels}
    "images_subdir": "train_dataset",
    "labels_subdir": "labels",
    "json_merge_labels": False,       # labelme 里出现多个 label 名时，True=全部并成同一前景类
    "img_size": [288, 512],           # [H, W]，需为 32 的倍数。AIDK 常用 288,512；labelme 数据用 224,224 或 64,64
    "in_channels": 1,                 # 1=灰度，3=RGB
    "num_classes": 0,                 # 含背景；0=自动（labelme 单类别 -> 2，AIDK 三类 -> 3）
    "mean": [127.5],                  # 归一化 (x-mean)/std，0~255 单位；RKNN 转换会读同一组值
    "std": [127.5],
    "val_ratio": 0.2,
    "seed": 42,
    "cache": True,                    # 预先把全部图缩放后放进内存（大图数据集强烈建议开）
    # ---- 模型 ----
    "arch": "unet",                   # unet / fpn / linknet / pan / unetplusplus / manet / deeplabv3plus ...
    "encoder": "resnet18",            # resnet18 / resnet34 / mobilenet_v2 / tu-xxx(timm 模型) ...
    "encoder_weights": "imagenet",    # imagenet 或 none（下载失败会自动退回 none 并提示）
    # ---- 训练 ----
    "epochs": 100,
    "batch_size": 8,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "loss": "ce_dice",                # ce / dice / ce_dice
    "amp": True,                      # 混合精度（仅 CUDA）
    "num_workers": 0,                 # 数据已缓存且增强很轻，Windows 下 0 最省事
    "device": "auto",                 # auto / cuda / cpu
    "vis_samples": 8,                 # 训练结束后保存多少张验证集可视化
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


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, amp):
    """返回 dict：loss、各类 dice/iou、前景平均 mdice/miou（不含背景类 0，不存在的类跳过）。"""
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
    conf = conf.view(num_classes, num_classes).double().cpu()  # 行=真值 列=预测
    tp = conf.diag()
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    nan = torch.tensor(float("nan"), dtype=torch.double)
    dice = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1), nan)
    iou = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn).clamp(min=1), nan)
    mdice, miou = float(torch.nanmean(dice[1:])), float(torch.nanmean(iou[1:]))
    # 验证集里前景类全都不存在时 nanmean 为 nan，按 0 处理，避免 best 判断失效
    mdice, miou = (0.0 if mdice != mdice else mdice), (0.0 if miou != miou else miou)
    return {"loss": loss_sum / max(n, 1), "dice": dice.tolist(), "iou": iou.tolist(), "mdice": mdice, "miou": miou}


def _clean(v):
    """NaN -> None，便于写 json"""
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


@torch.no_grad()
def save_vis(model, ds, meta, out_dir, device, n):
    """每张保存 [原图 | 真值叠加 | 预测叠加]"""
    if n <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i in np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int):
        img, mask = ds.get_raw(int(i))
        x = torch.from_numpy(normalize(img, meta["mean"], meta["std"]))[None].to(device)
        pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
        panel = np.hstack([to_bgr(img), overlay(img, mask), overlay(img, pred)])
        imwrite_unicode(out_dir / ("val_%03d.png" % i), panel)


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
    print("=" * 70)
    print("[run] %s | 类别数 %d | 设备 %s | 输出 %s" % (name, C, device, run_dir))

    ew = None if str(cfg["encoder_weights"]).lower() in ("none", "") else cfg["encoder_weights"]
    model = build_model(cfg["arch"], cfg["encoder"], cfg["in_channels"], C, ew).to(device)

    meta = {  # 只放基础类型，保证 torch.load 在新版本下也安全
        "arch": cfg["arch"], "encoder": cfg["encoder"], "in_channels": int(cfg["in_channels"]),
        "num_classes": int(C), "img_size": [hw[0], hw[1]],
        "mean": [float(v) for v in cfg["mean"]], "std": [float(v) for v in cfg["std"]],
        "dataset": cfg["dataset"], "data_root": str(cfg["data_root"]), "images_subdir": cfg["images_subdir"],
        "kind": info["kind"], "class_values": info.get("class_values"), "label2idx": info.get("label2idx"),
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

    best, best_epoch, best_metrics = -1.0, 0, None
    with open(run_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "mdice", "miou", "lr"] + ["dice_c%d" % c for c in range(C)])
        for ep in range(1, cfg["epochs"] + 1):
            model.train()
            tl, tn = 0.0, 0
            for x, y in train_loader:
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
            lr_now = opt.param_groups[0]["lr"]
            sched.step()

            m = evaluate(model, val_loader, criterion, device, C, amp)
            w.writerow([ep, tl / tn, m["loss"], m["mdice"], m["miou"], lr_now] + m["dice"])
            f.flush()
            mark = ""
            if m["mdice"] > best:
                best, best_epoch, best_metrics = m["mdice"], ep, m
                torch.save({"model": model.state_dict(), "meta": meta}, run_dir / "best.pth")
                mark = " *best"
            print("ep %3d/%d  train %.4f  val %.4f  mDice(fg) %.4f  dice %s%s" % (
                ep, cfg["epochs"], tl / tn, m["loss"], m["mdice"], np.round(m["dice"], 3).tolist(), mark))
    torch.save({"model": model.state_dict(), "meta": meta}, run_dir / "last.pth")

    # 用最佳权重做可视化与效率统计
    ck = torch.load(str(run_dir / "best.pth"), map_location="cpu")
    model.load_state_dict(ck["model"])
    save_vis(model, val_ds, meta, run_dir / "vis", device, int(cfg["vis_samples"]))
    shape = (1, int(cfg["in_channels"]), hw[0], hw[1])
    results = {
        "name": name, "arch": cfg["arch"], "encoder": cfg["encoder"], "img_size": [hw[0], hw[1]],
        "num_classes": C, "epochs": cfg["epochs"], "best_epoch": best_epoch,
        "mdice": _clean(best_metrics["mdice"]), "miou": _clean(best_metrics["miou"]),
        "dice": _clean(best_metrics["dice"]), "iou": _clean(best_metrics["iou"]),
        "params_m": round(count_params(model), 3),
        "gflops": count_gflops(model, shape, device),
        "latency_ms": round(benchmark_ms(model, shape, device), 3),
        "latency_device": device.type, "train_minutes": round((time.time() - t0) / 60, 2),
        "run_dir": str(run_dir),
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[done] best mDice(fg)=%.4f @ep%d | %.2fM params | %s GFLOPs | %.2f ms(%s)" % (
        best, best_epoch, results["params_m"], results["gflops"], results["latency_ms"], device.type))
    return results


if __name__ == "__main__":  # Windows 下启用多进程 DataLoader 必须有这个保护
    run(parse_args(DEFAULTS, "训练单个分割模型"))
