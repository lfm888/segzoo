"""探测 data/ 下所有数据集的标签形式（json / png 单通道索引 / png 多通道 one-hot）、类别、数量与大小。

直接运行：python inspect_data.py
覆盖参数：python inspect_data.py --data_root data --max_scan 200
每个数据集打印一块详情，最后打印一张总览表。
"""
import json
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO

import cv2
import numpy as np

from dataset import (INFO_FILE, _align_names, collect_pairs, detect_mask_mode, find_split_csv,
                     guess_names_from_json, read_class_names, read_label_info_file, read_mask,
                     read_split_csv)
from utils import abspath, imread_unicode, parse_args

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "data_root": "data",         # 数据集根目录（每个子目录是一个数据集）
    "images_subdir": "train_dataset",
    "labels_subdir": "labels",
    "max_scan": 200,             # 每个数据集最多抽样多少张 mask 统计像素值/通道（0=全部）
    "max_probe": 8,              # 最多探测多少张图像的分辨率/通道
    "split_csv": "auto",         # auto / none，是否检查数据集里的划分 csv
}
# ===========================================================================

W = 84


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0


def dir_size(path):
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total

def sample_list(items, k):
    if k <= 0 or len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).astype(int)
    return [items[i] for i in idx]


def quiet(fn, *a, **k):
    """调用 dataset.py 里的辅助函数但不让它打日志（保持报告的版式干净）"""
    with redirect_stdout(StringIO()):
        return fn(*a, **k)


def probe_image(path):
    """快速取分辨率/通道数：优先用 PIL 只读文件头，失败再退回 cv2。"""
    try:
        from PIL import Image
        with Image.open(str(path)) as im:
            w, h = im.size
            ch = len(im.getbands())
            return h, w, ch
    except Exception:
        img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        ch = 1 if img.ndim == 2 else img.shape[2]
        return img.shape[0], img.shape[1], ch


def _section(title):
    tag = "[%s]" % title
    pad = max(4, W - _dwidth(tag) - 4)
    print("-" * 4 + " " + tag + " " + "-" * pad)


def _dwidth(s):
    """显示宽度：中文字符按 2 列"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))


def _pad(s, width):
    """按显示宽度左对齐补空格"""
    s = str(s)
    return s + " " * max(0, width - _dwidth(s))


def _kv(lines, k, v):
    lines.append("  %s%s" % (_pad(k, 16), v))


# --------------------------------------------------------------------------
# 图像 / 标签统计
# --------------------------------------------------------------------------
def describe_images(imgs, max_probe):
    exts = Counter(p.suffix.lower() for p in imgs)
    probes = [probe_image(p) for p in sample_list(imgs, max_probe)]
    shapes = Counter((h, w) for h, w, _ in probes)
    chans = sorted({c for _, _, c in probes})
    if len(shapes) == 1:
        (h, w), _ = shapes.most_common(1)[0]
        size_txt = "%dx%d" % (w, h)
        if len(probes) < len(imgs):
            size_txt += "（抽样 %d 张）" % len(probes)
    else:
        size_txt = "多种: " + ", ".join("%dx%d" % (w, h) for (h, w), _ in shapes.most_common(4))
        size_txt += "（抽样 %d 张）" % len(probes)
    ch_txt = "/".join("%d 通道" % c for c in chans) + ("（灰度或彩色混合）" if len(chans) > 1 else "")
    ext_txt = ", ".join("%s×%d" % (e, c) for e, c in exts.most_common())
    return {"n": len(imgs), "ext": ext_txt, "size": size_txt, "ch": ch_txt,
            "hw": shapes.most_common(1)[0][0] if len(shapes) == 1 else None}


def describe_mask_labels(pairs, root, max_scan):
    """分析 mask 标签：形式、类别数、类名、各类像素占比、取值。"""
    lab_ext = pairs[0][1].suffix.lower()
    res = {"file": lab_ext, "form": "", "n_classes": 0, "names": [], "ratio": None,
           "values": None, "channels": None, "dtype": None, "same_size": None}
    if lab_ext == ".json":
        names = set()
        for _, lp in pairs:
            try:
                with open(lp, "r", encoding="utf-8") as f:
                    for s in json.load(f).get("shapes", []):
                        if s.get("label") is not None:
                            names.add(str(s["label"]))
            except (OSError, ValueError):
                continue
        res["form"] = "json（labelme 多边形）"
        res["short"] = "json(labelme)"
        res["n_classes"] = len(names) + 1
        res["names"] = ["background"] + sorted(names)
        return res

    mode = detect_mask_mode(pairs, "auto")
    picked = sample_list(pairs, max_scan)
    dtype, ch = None, None
    same_size, checked = True, 0
    if mode == "index":
        counts, total = Counter(), 0
        for ip, lp in picked:
            m = read_mask(lp)
            dtype = dtype or str(m.dtype)
            if m.ndim == 3:
                m = m[..., 0]
            if checked < 4:
                h, w, _ = probe_image(ip)
                same_size = same_size and (m.shape[0], m.shape[1]) == (h, w)
                checked += 1
            vals, cnts = np.unique(m, return_counts=True)
            counts.update({int(v): int(c) for v, c in zip(vals, cnts)})
            total += int(m.size)
        values = sorted(counts)
        res["dtype"] = dtype
        res["values"] = values
        # 归类成“二值”还是“多值索引”
        if len(values) == 2 and values[0] == 0 and values[1] in (1, 255):
            res["form"] = "png 单通道二值（0=背景，%d=前景）" % values[1]
            res["short"] = "png 单通道二值"
        else:
            res["form"] = "png 单通道索引图（同一通道不同像素值=不同类）"
            res["short"] = "png 单通道索引(%d 种值)" % len(values)
        bg_ok = 0 in values
        class_values = values if bg_ok else [0] + values
        res["n_classes"] = len(class_values)
        names = quiet(read_class_names, root) or quiet(guess_names_from_json, pairs, len(class_values))
        res["names"] = _align_names(names, len(class_values), "index")
        ratio = {}
        for v in values:
            idx = class_values.index(v)
            ratio[idx] = ratio.get(idx, 0) + counts[v] / max(total, 1)
        if not bg_ok:
            ratio[0] = 1.0 - sum(ratio.values())   # mask 里没出现的背景
        res["ratio"] = ratio
    else:
        n_ch = None
        fg = None
        overlap = 0
        total = 0
        for ip, lp in picked:
            m = read_mask(lp)
            dtype = dtype or str(m.dtype)
            n_ch = n_ch or m.shape[2]
            b = m > 0
            fg = b.reshape(-1, b.shape[2]).sum(0) if fg is None else fg + b.reshape(-1, b.shape[2]).sum(0)
            overlap += int((b.sum(2) > 1).sum())
            total += b.shape[0] * b.shape[1]
            if checked < 4:
                h, w, _ = probe_image(ip)
                same_size = same_size and (m.shape[0], m.shape[1]) == (h, w)
                checked += 1
        kept = [c for c in range(n_ch) if 0 < fg[c] < total]
        dropped = [c for c in range(n_ch) if c not in kept]
        res["dtype"], res["channels"] = dtype, n_ch
        res["n_classes"] = len(kept) + 1
        res["form"] = "png 多通道 one-hot（每通道=1 类，通道内 >0 为前景%s）" % (
            "，通道 %s 全背景/全前景已忽略" % dropped if dropped else "")
        res["short"] = "png 多通道 one-hot(%dch)" % len(kept)
        names = quiet(read_class_names, root) or quiet(guess_names_from_json, pairs, res["n_classes"])
        res["names"] = _align_names(names, res["n_classes"], "channels")
        ratio = {0: 1.0 - sum(fg[c] for c in kept) / max(total, 1)}
        for i, c in enumerate(kept):
            ratio[i + 1] = fg[c] / max(total, 1)
        res["ratio"] = ratio
        res["overlap"] = overlap / max(total, 1)
    res["same_size"] = same_size
    res["scanned"] = len(picked)
    return res


def describe_split(root, cfg, pairs):
    path = find_split_csv(root, cfg.get("split_csv", "auto"))
    if path is None:
        return None
    table = read_split_csv(path)
    cats = Counter(table.get(ip.name, ("(未列出)", None))[0] for ip, _ in pairs)
    return {"path": path, "counts": cats}


# --------------------------------------------------------------------------
# 每个数据集
# --------------------------------------------------------------------------
def describe_dataset(root, cfg):
    imgs = [p for p in sorted((root / cfg["images_subdir"]).iterdir())
            if p.is_file() and p.suffix.lower() in
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")] if (root / cfg["images_subdir"]).is_dir() else []
    out = {"name": root.name, "path": root, "err": None, "img": describe_images(imgs, cfg["max_probe"]) if imgs else None}
    try:
        pairs = quiet(collect_pairs, root / cfg["images_subdir"], root / cfg["labels_subdir"])
    except Exception as e:                                    # 目录缺失 / 无配对
        out["err"] = "%s: %s" % (type(e).__name__, e)
        return out

    lab_dir = pairs[0][1].parent
    lab_files = [p for p in lab_dir.iterdir() if p.is_file()]
    lab_bytes = sum(p.stat().st_size for p in lab_files)
    img_bytes = sum(p.stat().st_size for p in imgs)
    lab = describe_mask_labels(pairs, root, cfg["max_scan"])
    lab["n_files"] = len(lab_files)
    lab["bytes"] = lab_bytes
    out.update({"pairs": pairs, "lab": lab, "img_bytes": img_bytes, "lab_bytes": lab_bytes,
                "n_img": len(imgs), "split": describe_split(root, cfg, pairs),
                "total_bytes": dir_size(root)})
    return out


def print_dataset(i, n, d):
    print()
    _section("%d/%d  %s" % (i, n, d["name"]))
    _kv_line = []
    _kv(_kv_line, "路径", str(d["path"]))
    print("\n".join(_kv_line))
    if d["err"]:
        print("  错误            %s" % d["err"])
        return
    img, lab = d["img"], d["lab"]
    lines = []
    if img:
        _kv(lines, "图像", "%d 张  %s" % (img["n"], img["ext"]))
        _kv(lines, "图像尺寸", "%s   %s" % (img["size"], img["ch"]))
        _kv(lines, "图像大小", fmt_bytes(d["img_bytes"]))
    _kv(lines, "标签文件", "%d 个  %s   %s" % (lab["n_files"], lab["file"], fmt_bytes(lab["bytes"])))
    _kv(lines, "标签形式", lab["form"])
    if lab["dtype"]:
        extra = "dtype=%s" % lab["dtype"]
        if lab["channels"]:
            extra += "  通道数=%d" % lab["channels"]
        if lab["values"] is not None:
            vals = lab["values"]
            extra += "  像素值=%s" % (vals if len(vals) <= 8 else "%s ... (%d 种)" % (vals[:8], len(vals)))
        _kv(lines, "标签细节", extra)
    if lab["same_size"] is not None:
        _kv(lines, "尺寸一致性", "mask 与图像同尺寸" if lab["same_size"] else "不一致（会按最近邻缩放到 img_size）")
    _kv(lines, "类别数", "%d（含背景）" % lab["n_classes"])
    _kv(lines, "类别名", "  ".join("%d=%s" % (k, v) for k, v in enumerate(lab["names"])))
    if lab.get("ratio"):
        parts = []
        for k in sorted(lab["ratio"]):
            name = lab["names"][k] if k < len(lab["names"]) else "class_%d" % k
            parts.append("%s %.2f%%" % (name, 100 * lab["ratio"][k]))
        _kv(lines, "像素占比", "  ".join(parts))
    if lab.get("overlap"):
        _kv(lines, "通道重叠", "%.2f%% 的像素同时属于多个通道（转索引图时后通道覆盖前通道）" % (100 * lab["overlap"]))
    if lab.get("scanned"):
        _kv(lines, "统计抽样", "%d 张 mask" % lab["scanned"])
    _kv(lines, "配对", "%d/%d 张图像有标签%s" % (
        len(d["pairs"]), d["n_img"],
        "" if len(d["pairs"]) == d["n_img"] else "（缺 %d 张）" % (d["n_img"] - len(d["pairs"]))))
    if d["split"]:
        _kv(lines, "划分 csv", "%s   %s" % (
            d["split"]["path"].name, "  ".join("%s=%d" % (k, v) for k, v in d["split"]["counts"].most_common())))
    cache = read_label_info_file(d["path"] / INFO_FILE)
    if cache:
        _kv(lines, "扫描缓存", "%s（%s 生成）—— 训练时直接复用，不用再扫 %d 个标签文件" % (
            INFO_FILE, cache.get("generated", "?"), d["lab"]["n_files"]))
    else:
        _kv(lines, "扫描缓存", "无（首次训练时会自动生成 %s）" % INFO_FILE)
    _kv(lines, "磁盘合计", fmt_bytes(d["total_bytes"]))
    print("\n".join(lines))


def main():
    cfg = parse_args(DEFAULTS, "探测所有数据集的标签形式 / 类别 / 数量 / 大小")
    root = abspath(cfg["data_root"])
    if not root.is_dir():
        raise SystemExit("data_root 不存在: %s" % root)

    roots = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / cfg["images_subdir"]).is_dir() or (p / cfg["labels_subdir"]).is_dir():
            roots.append(p)
        else:
            print("[skip] %s 不是数据集目录（没有 %s/ 或 %s/）" % (p.name, cfg["images_subdir"], cfg["labels_subdir"]))

    print("=" * W)
    print("数据集探测  |  根目录 %s  |  共 %d 个数据集" % (root, len(roots)))
    print("=" * W)

    rows = []
    for i, p in enumerate(roots, 1):
        d = describe_dataset(p, cfg)
        print_dataset(i, len(roots), d)
        rows.append(d)

    # ---------------- 总览表 ----------------
    print()
    print("=" * W)
    print("总览")
    print("-" * W)
    cols = (22, 12, 28, 6, 14)
    print("  " + _pad("数据集", cols[0]) + _pad("配对/图像", cols[1]) + _pad("标签形式", cols[2])
          + _pad("类别", cols[3]) + _pad("输入尺寸", cols[4]) + "磁盘")
    generic = []
    for d in rows:
        if d["err"]:
            print("  " + _pad(d["name"], cols[0]) + "读取失败: " + d["err"])
            continue
        lab, img = d["lab"], d["img"]
        size = img["size"].split("（")[0] if img else "-"
        print("  " + _pad(d["name"], cols[0]) + _pad("%d/%d" % (len(d["pairs"]), d["n_img"]), cols[1])
              + _pad(lab.get("short", lab["form"]), cols[2]) + _pad(lab["n_classes"], cols[3])
              + _pad(size, cols[4]) + fmt_bytes(d["total_bytes"]))
        if any(str(x).startswith("class_") for x in lab["names"][1:]):
            generic.append(d["name"])
    print("-" * W)
    print("  合计磁盘 %s   |   标签形式: json=labelme 多边形；png 单通道索引=像素值代表类别；"
          "png 多通道 one-hot=每通道一类" % fmt_bytes(sum(d["total_bytes"] for d in rows if not d["err"])))
    if generic:
        print("  提示: %s 没有 classes.txt / labelme json 可推断类名，目前显示为 class_1..N" % ", ".join(generic))
        print("        在该数据集目录放 classes.txt（每行一个类名，可含 background）即可得到稳定的类名与索引顺序")
    print("=" * W)


if __name__ == "__main__":
    main()
