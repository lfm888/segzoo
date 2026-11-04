"""数据集：同时支持 labelme(json 多边形) 与 mask 图像（png 等）两种标签。

目录约定：<data_root>/<dataset>/train_dataset/ 放图像，<data_root>/<dataset>/labels/ 放标签。
  - 标签是 .json  -> 按 labelme 多边形栅格化；类别 = 背景(0) + 每个 label 名(1..N)
  - 标签是图像    -> 自动扫描全部 mask 的像素值，按从小到大映射为连续类别 0..K-1
                     （例如 {0,128,255} -> {0,1,2}），映射关系会打印出来请核对
图像与标签按文件名主干匹配（也接受 xxx_mask / xxx_label / xxx_gt 等后缀）。
"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import IMG_EXTS, abspath, imread_unicode, load_image, normalize, resize_image, to_hw

MASK_SUFFIXES = ["", "_mask", "_label", "_labels", "_gt", "_seg", "-mask", "-label"]


# --------------------------------------------------------------------------
# 配对 & 标签信息扫描
# --------------------------------------------------------------------------
def collect_pairs(img_dir, lab_dir):
    if not img_dir.is_dir():
        raise FileNotFoundError("图像目录不存在: %s" % img_dir)
    if not lab_dir.is_dir():
        raise FileNotFoundError("标签目录不存在: %s" % lab_dir)

    lab_index = {}
    for p in sorted(lab_dir.iterdir()):
        if p.is_file() and (p.suffix.lower() in IMG_EXTS or p.suffix.lower() == ".json"):
            lab_index.setdefault(p.stem, []).append(p)

    pairs, missing = [], []
    for ip in sorted(img_dir.iterdir()):
        if not (ip.is_file() and ip.suffix.lower() in IMG_EXTS):
            continue
        lp = None
        for suf in MASK_SUFFIXES:
            cand = lab_index.get(ip.stem + suf)
            if cand:
                # 同名同时有 json 和图像时优先 json
                lp = sorted(cand, key=lambda q: q.suffix.lower() != ".json")[0]
                break
        if lp is None:
            missing.append(ip.name)
        else:
            pairs.append((ip, lp))

    if missing:
        print("[warn] %d 张图像找不到对应标签，已忽略（示例: %s）" % (len(missing), missing[:3]))
    if not pairs:
        raise RuntimeError("没有匹配到任何 图像-标签 对，请检查目录 %s 与 %s" % (img_dir, lab_dir))
    kinds = {lp.suffix.lower() == ".json" for _, lp in pairs}
    if len(kinds) > 1:
        raise RuntimeError("标签目录里 json 与图像标签混用，请统一成一种")
    return pairs


def read_mask(path):
    m = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:  # 伪彩/多通道 mask：取第 0 通道（若是彩色编码的 mask 请先转成灰度索引图）
        m = m[..., 0]
    return m


def scan_label_info(pairs, merge_json_labels=False):
    if pairs[0][1].suffix.lower() == ".json":
        names = set()
        for _, lp in pairs:
            with open(lp, "r", encoding="utf-8") as f:
                for s in json.load(f).get("shapes", []):
                    if s.get("label") is not None:
                        names.add(s["label"])
        names = sorted(names)
        if merge_json_labels:
            label2idx = {n: 1 for n in names}
            num_classes = 2
        else:
            label2idx = {n: i + 1 for i, n in enumerate(names)}
            num_classes = len(names) + 1
        info = {"kind": "json", "label2idx": label2idx, "num_classes": num_classes}
        print("[data] labelme 标签名 -> 类别索引: %s（0=背景）" % label2idx)
    else:
        vals = set()
        for _, lp in pairs:
            vals.update(np.unique(read_mask(lp)).tolist())
        values = sorted(int(v) for v in vals)
        info = {"kind": "mask", "class_values": values, "num_classes": len(values)}
        print("[data] mask 像素值 -> 类别索引: %s（最小值视为背景 0）" % {v: i for i, v in enumerate(values)})
    return info


def rasterize_labelme(json_path, label2idx, fallback_hw):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    H = int(d.get("imageHeight") or fallback_hw[0])
    W = int(d.get("imageWidth") or fallback_hw[1])
    mask = np.zeros((H, W), np.uint8)
    for s in d.get("shapes", []):
        idx = label2idx.get(s.get("label"))
        if idx is None:
            continue
        pts = np.array(s["points"], dtype=np.float32)
        st = s.get("shape_type", "polygon")
        if st == "polygon" and len(pts) >= 3:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], int(idx))
        elif st == "rectangle" and len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            cv2.rectangle(mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), int(idx), -1)
        elif st == "circle" and len(pts) >= 2:
            c, e = pts[0], pts[1]
            cv2.circle(mask, (int(round(c[0])), int(round(c[1]))), int(round(float(np.linalg.norm(e - c)))), int(idx), -1)
    return mask


# --------------------------------------------------------------------------
# 数据增强（只用 cv2/numpy；图像双线性、标签最近邻，保证二者几何变换一致）
# 外观增强（对比度/亮度/gamma/噪声/模糊）主要用于缓解跨设备成像差异（域偏移）
# --------------------------------------------------------------------------
def augment(img, mask, a):
    h, w = mask.shape[:2]
    if random.random() < a["hflip"]:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if random.random() < a["vflip"]:
        img, mask = img[::-1], mask[::-1]
    img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

    if random.random() < a["affine_p"]:
        ang = random.uniform(-a["rotate"], a["rotate"])
        sc = random.uniform(1 - a["scale"], 1 + a["scale"])
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc)
        M[0, 2] += random.uniform(-a["shift"], a["shift"]) * w
        M[1, 2] += random.uniform(-a["shift"], a["shift"]) * h
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

    img = img.astype(np.float32)
    if random.random() < a["color_p"]:
        mu = img.mean()
        alpha = 1.0 + random.uniform(-a["contrast"], a["contrast"])
        beta = random.uniform(-a["brightness"], a["brightness"]) * 255.0
        img = (img - mu) * alpha + mu + beta
        img = np.clip(img, 0, 255)
        g = random.uniform(1.0 / a["gamma"], a["gamma"])
        img = 255.0 * np.power(img / 255.0, g)
        if a["noise"] > 0:
            img = img + np.random.randn(*img.shape).astype(np.float32) * random.uniform(0, a["noise"]) * 255.0
        img = np.clip(img, 0, 255)
    if random.random() < a["blur_p"]:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img, mask


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class SegDataset(Dataset):
    def __init__(self, pairs, info, img_size, in_channels, mean, std, aug=None, cache=True):
        self.pairs, self.info = pairs, info
        self.hw = to_hw(img_size)
        self.in_channels, self.mean, self.std, self.aug = in_channels, mean, std, aug
        self._cache = [None] * len(pairs)
        if cache:  # 先一次性解码+缩放到目标尺寸放内存，训练时不再反复解码大图
            for i in range(len(pairs)):
                self._cache[i] = self._load(i)

    def _load(self, i):
        ip, lp = self.pairs[i]
        img = load_image(ip, self.in_channels)
        if self.info["kind"] == "json":
            mask = rasterize_labelme(lp, self.info["label2idx"], img.shape[:2])
        else:
            mask = np.searchsorted(np.array(self.info["class_values"]), read_mask(lp)).astype(np.uint8)
        return resize_image(img, self.hw), resize_image(mask, self.hw, is_mask=True)

    def get_raw(self, i):
        """返回已缩放到目标尺寸的 (uint8 图像, 类别索引 mask)，用于可视化。"""
        return self._cache[i] if self._cache[i] is not None else self._load(i)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img, mask = self.get_raw(i)
        if self.aug:
            img, mask = augment(img, mask, self.aug)
        x = normalize(img, self.mean, self.std)
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64))


def build_datasets(cfg):
    """按 cfg 构建 (train_ds, val_ds, info)。划分由 seed 决定，同一数据集上不同模型的验证集完全相同。"""
    root = abspath(cfg["data_root"]) / cfg["dataset"]
    pairs = collect_pairs(root / cfg["images_subdir"], root / cfg["labels_subdir"])
    info = scan_label_info(pairs, cfg["json_merge_labels"])
    if len(pairs) < 2:
        raise RuntimeError("至少需要 2 张图像才能划分训练/验证集")

    idx = list(range(len(pairs)))
    random.Random(cfg["seed"]).shuffle(idx)
    n_val = min(len(idx) - 1, max(1, int(round(len(idx) * cfg["val_ratio"]))))
    val_idx, train_idx = sorted(idx[:n_val]), sorted(idx[n_val:])
    print("[data] 共 %d 对，训练 %d / 验证 %d" % (len(pairs), len(train_idx), len(val_idx)))

    aug = {k[4:]: cfg[k] for k in cfg if k.startswith("aug_")}
    common = dict(info=info, img_size=cfg["img_size"], in_channels=cfg["in_channels"],
                  mean=cfg["mean"], std=cfg["std"], cache=cfg["cache"])
    train_ds = SegDataset([pairs[i] for i in train_idx], aug=aug, **common)
    val_ds = SegDataset([pairs[i] for i in val_idx], aug=None, **common)
    info = dict(info, val_files=[pairs[i][0].name for i in val_idx])
    return train_ds, val_ds, info
