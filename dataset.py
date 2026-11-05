"""数据集：同时支持 labelme(json 多边形) 与 mask 图像（png 等）两种标签。

目录约定：<data_root>/<dataset>/train_dataset/ 放图像，<data_root>/<dataset>/labels/ 放标签。
  - 标签是 .json  -> 按 labelme 多边形栅格化；类别 = 背景(0) + 每个 label 名(1..N)
  - 标签是图像    -> 两种编码都支持（mask_mode="auto" 自动判断）：
      "index"    单通道索引图：像素值 = 类别值，按从小到大映射为连续类别 0..K-1
                 （例如 {0,128,255} -> {0,1,2}）。像素值种类过多（>64）时按二值 mask 处理（>0=前景）。
      "channels" 多通道 mask：每个通道 = 一个类别，通道内 >0 即前景，通道之间允许重叠；
                 与 AIDK 数据集一致（R=Cornea, G=Iris, B=Lesion）。转成索引图时
                 “后面的通道覆盖前面的通道”，所以把更具体的类放在后面的通道（Lesion 盖住 Cornea）。
类别命名（可选，仅影响打印 / meta，不影响训练）：按下面顺序取第一个可用的：
  1) train.py 的 class_names 参数
  2) <dataset>/classes.txt 或 <dataset>/labels/classes.txt（每行一个类名，可含 background）
     或 classes.json（["background","Cornea",...]）
  3) 图像目录/标签目录里的 labelme .json 标签名（如 inner_circle_dataset 的 'center'，需个数与 mask 类别数一致）
  4) 默认名 class_1, class_2 ...
训练/验证划分：默认随机（seed 决定）；若存在 <dataset>/split/test_train_data.csv
  （image_id,category 两列，可选 fold 列），则按它划分（见 split_by_csv / train.py 的 split_csv 参数）。
没有 labels 目录时，自动回退为把图像目录里的 labelme .json 当标签用。
图像与标签按文件名主干匹配（也接受 xxx_mask / xxx_label / xxx_gt 等后缀）。
"""
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import IMG_EXTS, abspath, imread_unicode, load_image, normalize, resize_image, to_hw

MASK_SUFFIXES = ["", "_mask", "_label", "_labels", "_gt", "_seg", "-mask", "-label"]
CLASS_FILES = ("classes.txt", "classes.json")  # 可选的类别名文件
SPLIT_CSV_CANDIDATES = ("split/test_train_data.csv", "test_train_data.csv", "split/train_val_test.csv")
MAX_INDEX_CLASSES = 64   # 索引图里超过这么多种像素值，就当成二值 mask（>0=前景）
INFO_FILE = "label_info.txt"   # 标签扫描缓存（自动生成）
INFO_VERSION = 1
BG_NAMES = ("background", "bg", "backgrounds", "背景", "0", "none", "unlabeled")


# --------------------------------------------------------------------------
# 配对
# --------------------------------------------------------------------------
def collect_pairs(img_dir, lab_dir):
    if not img_dir.is_dir():
        raise FileNotFoundError("图像目录不存在: %s" % img_dir)
    same_dir = False
    if not lab_dir.is_dir():
        # 兼容“标签还没转成 mask”的数据集：labelme 的 .json 直接放在图像目录里
        jsons = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json")
        if not jsons:
            raise FileNotFoundError("标签目录不存在: %s" % lab_dir)
        print("[data] 标签目录 %s 不存在，改用图像目录里的 %d 个 labelme .json 作为标签"
              % (lab_dir, len(jsons)))
        lab_dir, same_dir = img_dir, True
    elif lab_dir.resolve() == img_dir.resolve():
        same_dir = True

    # 标签与图像同一目录时只有 .json 能当标签（否则图像会跟自己对上）
    lab_exts = {".json"} if same_dir else (IMG_EXTS | {".json"})
    lab_index = {}
    for p in sorted(lab_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in lab_exts:
            lab_index.setdefault(p.stem, []).append(p)

    pairs, missing = [], []
    for ip in sorted(img_dir.iterdir()):
        if not (ip.is_file() and ip.suffix.lower() in IMG_EXTS):
            continue
        lp = None
        for suf in MASK_SUFFIXES:
            cand = [q for q in lab_index.get(ip.stem + suf, []) if q != ip]
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
    """读取 mask 原图，不做通道裁剪：返回 HxW（单通道）或 HxWxC（多通道）。"""
    m = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    return m


# --------------------------------------------------------------------------
# 类别名（可选）
# --------------------------------------------------------------------------
def read_class_names(root):
    """在 <dataset>/ 或 <dataset>/labels/ 下找 classes.txt / classes.json，返回类名列表（可为空）。"""
    for d in (root, root / "labels"):
        for fn in CLASS_FILES:
            p = d / fn
            if not p.is_file():
                continue
            if p.suffix.lower() == ".json":
                data = json.load(open(p, encoding="utf-8"))
                if isinstance(data, dict):
                    data = data.get("classes") or data.get("class_names")
                if not isinstance(data, list):
                    print("[warn] %s 格式不认识，应为类名列表，已忽略" % p)
                    continue
                names = [str(x).strip() for x in data]
            else:
                names = [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines()]
            names = [n for n in names if n and not n.startswith("#")]
            if names:
                print("[data] 类别名文件: %s -> %s" % (p, names))
                return names
    return []


def _align_names(names, num_classes, what):
    """把类名列表对齐到 num_classes（含背景 0）。不足则补默认名，多则截断并提示。"""
    default = ["background"] + ["class_%d" % i for i in range(1, num_classes)]
    if not names:
        return default
    names = list(names)
    if len(names) == num_classes - 1 and names[0].lower() not in BG_NAMES:
        names = ["background"] + names  # classes.txt 只写了前景类名
    if len(names) != num_classes:
        print("[warn] 类名 %d 个，与类别数 %d 不一致（%s），按顺序对齐" % (len(names), num_classes, what))
        names = (names + default)[:num_classes]
    return names


def guess_names_from_json(pairs, num_classes):
    """没有 classes.txt 时，从图像/标签目录里的 labelme .json 推断类名。
    典型场景：inner_circle_dataset 的 train_dataset/ 里同时放着原始 json（标签名 center），
    而 labels/ 里是转好的二值 mask，没它就只能叫 class_1。"""
    stems = {ip.stem for ip, _ in pairs}
    jsons = [p for ip, _ in pairs for p in [ip.parent / (ip.stem + ".json")] if p.is_file()]
    if not jsons:
        lab_dir = pairs[0][1].parent
        jsons = [p for p in sorted(lab_dir.glob("*.json")) if p.stem in stems]
    if not jsons:
        return []
    names = set()
    for p in jsons[:500]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for s in json.load(f).get("shapes", []):
                    if s.get("label") is not None:
                        names.add(str(s["label"]))
        except (OSError, ValueError):
            continue
    names = sorted(names)
    if len(names) == num_classes - 1:
        print("[data] 无 classes.txt，从 labelme json 推断类名: ['background'] + %s" % names)
        return ["background"] + names
    if names:
        print("[warn] labelme json 里有 %d 个类名 %s，与 mask 类别数 %d 不匹配，只用通道/像素值命名"
              % (len(names), names, num_classes))
    return []


# --------------------------------------------------------------------------
# 可选：CSV 划分文件（参考 python_model_train：<dataset>/split/test_train_data.csv）
#   image_id,category  -> image_id 是带扩展名的文件名，category=train/test/val
#   若还有 fold 列，则 fold==0 作为验证集（对应参考项目里的 GroupKFold 第 0 折）
# --------------------------------------------------------------------------
def find_split_csv(root, value):
    """value: auto / none / 具体路径。auto 时在数据集目录下找约定位置的 csv。"""
    if value is None or str(value).lower() in ("", "none", "false", "0"):
        return None
    if str(value).lower() != "auto":
        p = abspath(str(value))
        if not p.is_file():
            raise FileNotFoundError("split_csv 不存在: %s" % p)
        return p
    for rel in SPLIT_CSV_CANDIDATES:
        if (root / rel).is_file():
            return root / rel
    return None


def read_split_csv(path):
    """返回 {文件名: (category, fold)}；列名容错常见写法。"""
    import csv as _csv

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise RuntimeError("split csv 是空的: %s" % path)
    cols = {(k or "").strip().lower(): k for k in rows[0]}
    id_col = next((cols[k] for k in ("image_id", "image", "filename", "file", "name", "id") if k in cols), None)
    cat_col = next((cols[k] for k in ("category", "split", "set", "subset") if k in cols), None)
    fold_col = cols.get("fold")
    if id_col is None or cat_col is None:
        raise RuntimeError("split csv 需要 image_id 与 category 两列，实际列: %s" % list(cols.values()))
    out = {}
    for r in rows:
        name = (r.get(id_col) or "").strip()
        if not name:
            continue
        fold = r.get(fold_col) if fold_col else None
        try:
            fold = int(float(fold))
        except (TypeError, ValueError):
            fold = None
        out[name] = ((r.get(cat_col) or "").strip().lower(), fold)
    return out


# --------------------------------------------------------------------------
# mask 标签编码方式判断 + 类别扫描
# --------------------------------------------------------------------------
def _is_binary_channel(ch):
    """单通道取值种类 <=2（全 0 或 {0,255}）视为“该通道=一类”的二值通道。"""
    u = np.unique(ch)
    return len(u) <= 2


def detect_mask_mode(pairs, mask_mode="auto"):
    """判断 mask 标签的编码方式，返回 "index" 或 "channels"。"""
    if mask_mode in ("index", "channels"):
        return mask_mode
    probe = [pairs[i][1] for i in
             np.linspace(0, len(pairs) - 1, min(len(pairs), 8)).astype(int)]
    votes = {"index": 0, "channels": 0}
    for lp in probe:
        m = read_mask(lp)
        if m.ndim == 2 or m.shape[2] == 1:
            votes["index"] += 1
            continue
        if all(_is_binary_channel(m[..., c]) for c in range(m.shape[2])):
            votes["channels"] += 1          # 每通道只有 {0,255}：多通道多标签 mask（AIDK 风格）
        elif all(np.array_equal(m[..., 0], m[..., c]) for c in range(1, m.shape[2])):
            votes["index"] += 1             # 三通道完全相同：其实就是单通道索引图
        else:
            raise RuntimeError(
                "无法判断 mask 编码方式: %s\n"
                "  支持 1) 单通道索引图（像素值=类别值）\n"
                "      2) 多通道二值 mask（每通道=1类，取值 0/255）\n"
                "  若你的 mask 是彩色伪彩图，请先转成上面两种之一，或手动指定 mask_mode=index/channels"
                % lp)
    return "channels" if votes["channels"] >= votes["index"] else "index"


def _scan_channel_masks(pairs, names):
    """多通道 mask：类别数 = 通道数 + 1(背景)，统计通道间重叠比例。
    全背景 / 全前景的通道（典型是 RGBA 的 alpha 通道）会被自动忽略。"""
    n_ch = None
    total = 0
    fg_px = None
    multi_all = 0                     # 所有通道（含待剔除的）之间的重叠像素
    for _, lp in pairs:
        m = read_mask(lp)
        if m.ndim != 3:
            raise RuntimeError("mask 通道数不一致（%s 是单通道），请统一编码方式" % lp)
        if n_ch is None:
            n_ch, fg_px = m.shape[2], [0] * m.shape[2]
        elif m.shape[2] != n_ch:
            raise RuntimeError("mask 通道数不一致: 期望 %d 通道，%s 是 %d 通道" % (n_ch, lp, m.shape[2]))
        fg = m > 0
        for c in range(n_ch):
            fg_px[c] += int(fg[..., c].sum())
        if n_ch > 1:
            multi_all += int((fg.sum(2) > 1).sum())
        total += fg.shape[0] * fg.shape[1]

    # 全 0 或全 1 的通道不可能是一个真实类别（多为 alpha / 占位通道），剔除
    kept = [c for c in range(n_ch) if 0 < fg_px[c] < total]
    dropped = [c for c in range(n_ch) if c not in kept]
    if dropped:
        print("[warn] 通道 %s 在全部 mask 里都是全背景或全前景（多为 alpha/占位通道），已忽略" % dropped)
    if not kept:
        raise RuntimeError("所有 mask 通道都是全背景或全前景，无法作为类别通道")

    if not dropped:
        overlap = multi_all
    else:  # 剔除了通道，只统计真正参与分类的通道之间的重叠
        overlap = 0
        for _, lp in pairs:
            fg = read_mask(lp) > 0
            if len(kept) > 1:
                overlap += int((fg[..., kept].sum(2) > 1).sum())

    num_classes = len(kept) + 1
    if not names:
        names = guess_names_from_json(pairs, num_classes)
    cls_names = _align_names(names, num_classes, "channels")
    print("[data] mask 编码: 多通道（每通道=1 类），类别数 %d（含背景）:" % num_classes)
    for i, c in enumerate(kept):
        print("         通道 %d -> 类别 %d (%s)" % (c, i + 1, cls_names[i + 1]))
    print("         类别 0  = 背景 (%s)" % cls_names[0])
    if total and overlap:
        print("         通道重叠像素 %.2f%%：转索引图时“后通道覆盖前通道”（更具体的类放后面）"
              % (100.0 * overlap / total))
    return {"kind": "mask", "mask_mode": "channels", "num_classes": num_classes,
            "class_names": cls_names, "class_values": None, "channels": int(n_ch),
            "channel_map": [int(c) for c in kept],
            "overlap_ratio": round(overlap / total, 4) if total else 0.0}


def _scan_index_masks(pairs, names):
    """单通道索引图：收集所有像素值，按升序映射成连续类别；值太多则当二值 mask。"""
    vals = set()
    for k, (_, lp) in enumerate(pairs, 1):
        m = read_mask(lp)
        if m.ndim == 3:
            m = m[..., 0]
        vals.update(np.unique(m).tolist())
        if len(pairs) > 400 and k % 200 == 0:
            print("       扫描 mask %d/%d ..." % (k, len(pairs)))

    values = sorted(int(v) for v in vals)
    binary = False
    if len(values) > MAX_INDEX_CLASSES:
        print("[warn] mask 像素值有 %d 种（>%d），按二值 mask 处理：0=背景，>0=前景"
              % (len(values), MAX_INDEX_CLASSES))
        binary, class_values = True, [0, 1]
    elif 0 in values:
        class_values = values                      # 0 已经是背景
    else:
        class_values = [0] + values                # mask 里没有 0：补一个背景类，非零值各成一类

    num_classes = len(class_values)
    if not names:
        names = guess_names_from_json(pairs, num_classes)
    cls_names = _align_names(names, num_classes, "index")
    mapping = {0: 0, ">0": 1} if binary else {v: i for i, v in enumerate(class_values)}
    print("[data] mask 编码: %s，类别数 %d（含背景）；像素值 -> 类别索引: %s" % (
        "二值（>0=前景）" if binary else "单通道索引图", num_classes, mapping))
    print("         类别名 -> 索引: %s" % {n: i for i, n in enumerate(cls_names)})
    return {"kind": "mask", "mask_mode": "index", "num_classes": num_classes,
            "class_names": cls_names, "class_values": class_values, "binary": binary}


# --------------------------------------------------------------------------
# 标签扫描缓存：<dataset>/label_info.txt（自动生成，手改/删除都安全）
#   首次扫描要解码全部 mask（AIDK 768 张约 18s），结果存下来以后直接复用。
#   文件里记了标签目录的指纹（文件数/总字节/最大 mtime），标签改了会自动重扫。
# --------------------------------------------------------------------------
def label_fingerprint(pairs):
    lab_dir = pairs[0][1].parent
    n, total, mx = 0, 0, 0
    for p in lab_dir.iterdir():
        if p.is_file():
            try:
                st = p.stat()
            except OSError:
                continue
            n += 1
            total += st.st_size
            mx = max(mx, st.st_mtime_ns)
    return {"files": n, "bytes": total, "mtime": int(mx), "pairs": len(pairs)}


def read_label_info_file(path):
    """读 key = value（value 为 JSON）格式的缓存，异常/格式不对就当作没有缓存。"""
    if not path.is_file():
        return None
    try:
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            try:
                out[k.strip()] = json.loads(v.strip())
            except ValueError:
                out[k.strip()] = v.strip()
        return out
    except OSError:
        return None


def write_label_info_file(path, info, fp, mask_mode, merge_json_labels):
    """把扫描结果（含指纹）写成人类可读的 txt。"""
    keep = {"kind", "mask_mode", "num_classes", "class_names", "class_values", "label2idx",
            "channel_map", "channels", "binary", "overlap_ratio", "merged"}
    lines = ["# segzoo 标签扫描缓存：自动生成，删除后下次训练会重新扫描（手改 classes.txt 优先）",
             "version = %d" % INFO_VERSION,
             "generated = %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "json_merge_labels = %s" % json.dumps(bool(merge_json_labels)),
             "scan_mask_mode = %s" % json.dumps(str(mask_mode))]
    for k in sorted(keep):
        if k in info:
            lines.append("%s = %s" % (k, json.dumps(info[k], ensure_ascii=False)))
    lines.append("fingerprint = %s" % json.dumps(fp))
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as e:
        print("[warn] 标签扫描缓存写入失败（不影响训练）: %s" % e)


def apply_class_names(info, root, cfg_names=None):
    """最终确定类名：cfg 的 class_names > classes.txt/classes.json > 缓存/扫描结果。"""
    names = list(cfg_names) if cfg_names else read_class_names(root)
    if not names:
        return info
    if info["kind"] == "json":
        if info.get("merged"):
            return info
        found = list(info["label2idx"])
        fg = [n for n in names if n.lower() not in BG_NAMES]
        if fg:  # classes.txt 给出的顺序优先，保证类别索引稳定
            order = [n for n in fg if n in found] + [n for n in found if n not in fg]
            info["label2idx"] = {n: i + 1 for i, n in enumerate(order)}
            info["class_names"] = ["background"] + order
            print("[data] 按 classes.txt/参数 重排 labelme 类别顺序: %s" % info["label2idx"])
        return info
    info["class_names"] = _align_names(names, info["num_classes"], info.get("mask_mode") or "index")
    return info


def scan_label_info(pairs, merge_json_labels=False, mask_mode="auto", class_names=None,
                    root=None, use_cache=True):
    """扫描标签，返回 info：kind / mask_mode / num_classes / class_names / class_values / label2idx。
    结果缓存到 <dataset>/label_info.txt，下次直接复用（标签目录指纹变了才重扫）。"""
    root = Path(root) if root is not None else pairs[0][1].parent.parent   # <dataset>/
    cache_path = root / INFO_FILE
    fp = label_fingerprint(pairs)

    if use_cache and str(mask_mode) == "auto":              # 手动指定 mask_mode 时不读缓存
        cached = read_label_info_file(cache_path)
        if (cached and int(cached.get("version", 0)) == INFO_VERSION
                and cached.get("fingerprint") == fp
                and cached.get("json_merge_labels") == bool(merge_json_labels)
                and cached.get("kind") in ("mask", "json")):
            info = {k: cached[k] for k in
                    ("kind", "mask_mode", "num_classes", "class_names", "class_values",
                     "label2idx", "channel_map", "channels", "binary", "overlap_ratio", "merged") if k in cached}
            print("[data] 复用标签扫描缓存 %s（%s 类别 %d: %s）" % (
                cache_path.name, cached.get("generated", ""), info["num_classes"], info["class_names"]))
            info["from_cache"] = cache_path.name
            return apply_class_names(info, root, class_names)

    info = _scan_label_info(pairs, merge_json_labels, mask_mode, class_names, root)
    info["from_cache"] = None
    if str(mask_mode) == "auto":        # 强制指定 mask_mode 时不写入（避免污染 auto 的缓存）
        write_label_info_file(cache_path, info, fp, mask_mode, merge_json_labels)
        print("[data] 标签扫描结果已缓存: %s（下次训练直接复用）" % cache_path)
    return apply_class_names(info, root, class_names)


def _scan_label_info(pairs, merge_json_labels, mask_mode, class_names, root):
    """真正去解码标签的扫描（慢），结果会被 scan_label_info 缓存。"""
    names = list(class_names) if class_names else read_class_names(root)

    if pairs[0][1].suffix.lower() != ".json":
        if mask_mode in ("auto", "index", "channels"):
            return _scan_channel_masks(pairs, names) if detect_mask_mode(pairs, mask_mode) == "channels" \
                else _scan_index_masks(pairs, names)
        raise ValueError("mask_mode 只能是 auto / index / channels，当前: %r" % mask_mode)

    # ---- labelme json ----
    found = set()
    for _, lp in pairs:
        with open(lp, "r", encoding="utf-8") as f:
            for s in json.load(f).get("shapes", []):
                if s.get("label") is not None:
                    found.add(s["label"])
    # 类名文件给出的顺序优先（保证不同数据集/不同次运行的类别索引稳定）
    fg_hint = [n for n in names if n.lower() not in BG_NAMES]
    order = [n for n in fg_hint if n in found] + sorted(found - set(fg_hint))
    if merge_json_labels:
        label2idx = {n: 1 for n in order}
        num_classes = 2
        if len(order) > 1:
            print("[warn] json_merge_labels=True：%d 个标签名（%s）全部并成前景类 1" % (len(order), order))
    else:
        label2idx = {n: i + 1 for i, n in enumerate(order)}
        num_classes = len(order) + 1
    hint_fg = [n for n in names if n.lower() not in BG_NAMES]
    if hint_fg:
        cls_names = _align_names(hint_fg, num_classes, "json")
    elif merge_json_labels:
        cls_names = ["background", "foreground"]
    else:  # 没有 classes.txt 时直接用 labelme 里的标签名
        cls_names = ["background"] + order
        print("[data] 类别名: 未找到 classes.txt，直接使用 labelme 的标签名 %s" % order)
    print("[data] labelme 标签名 -> 类别索引: %s（0=背景）" % label2idx)
    return {"kind": "json", "mask_mode": None, "label2idx": label2idx, "merged": bool(merge_json_labels),
            "num_classes": num_classes, "class_names": cls_names, "class_values": None}


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


def mask_to_index(m, info):
    """把任意编码的 mask 转成单通道类别索引图（0=背景，1..K-1=前景类）。"""
    if m.ndim == 3:
        if info.get("mask_mode") == "channels":
            kept = info.get("channel_map") or list(range(m.shape[2]))
            fg = m > 0
            idx = np.zeros(fg.shape[:2], np.uint8)
            for i, c in enumerate(kept):
                if i + 1 >= info["num_classes"]:
                    break
                idx[fg[..., c]] = i + 1     # 后通道覆盖前通道
            return idx
        m = m[..., 0]
    if info.get("binary"):
        return (m > 0).astype(np.uint8)
    vals = np.asarray(info["class_values"], dtype=np.int64)
    idx = np.searchsorted(vals, m.astype(np.int64))
    return np.clip(idx, 0, len(vals) - 1).astype(np.uint8)


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
            mask = mask_to_index(read_mask(lp), self.info)
        return resize_image(img, self.hw), resize_image(mask, self.hw, is_mask=True)

    def get_raw(self, i):
        """返回已缩放到目标尺寸的 (uint8 图像, 类别索引 mask)，用于可视化。"""
        return self._cache[i] if self._cache[i] is not None else self._load(i)

    def get_full(self, i):
        """返回**原分辨率**的 (uint8 图像, 类别索引 mask)：给训练结束后的可视化用，
        与 predict.py 一样先把预测放大回原图尺寸再叠加，保证与原图完全对齐。"""
        ip, lp = self.pairs[i]
        img = load_image(ip, self.in_channels)
        if self.info["kind"] == "json":
            mask = rasterize_labelme(lp, self.info["label2idx"], img.shape[:2])
        else:
            mask = mask_to_index(read_mask(lp), self.info)
        if mask.shape[:2] != img.shape[:2]:   # 极端情况：mask 与图不同尺寸
            mask = resize_image(mask, img.shape[:2], is_mask=True)
        return img, mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img, mask = self.get_raw(i)
        if self.aug:
            img, mask = augment(img, mask, self.aug)
        x = normalize(img, self.mean, self.std)
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64))


def split_by_csv(pairs, csv_path, val_ratio, seed):
    """按 csv 里的 category / fold 列划分（参考 python_model_train 的 split/test_train_data.csv）。
    category 只能是 train 的行才参与训练；val/valid 行直接做验证集，test 行不参与；
    若还有 fold 列，则 fold==0 作为验证集（对应参考项目的 GroupKFold 第 0 折）。
    返回 (train_idx, val_idx, note)。"""
    table = read_split_csv(csv_path)
    matched = [(i, *table[ip.name]) for i, (ip, _) in enumerate(pairs) if ip.name in table]
    missing = len(pairs) - len(matched)
    cats = [c for _, c, _ in matched]
    print("[data] 使用 csv 划分: %s | 匹配 %d/%d 张%s" % (
        csv_path, len(matched), len(pairs),
        "（%d 张不在 csv 里，已忽略）" % missing if missing else ""))
    if cats:
        counts = {c: cats.count(c) for c in sorted(set(cats))}
        if len(counts) > 1 or "train" not in counts:
            print("         csv 里的 category 分布: %s" % counts)

    val_cats = {"val", "valid", "validation", "dev", "eval"}
    val_idx = [i for i, c, _ in matched if c in val_cats]
    if val_idx:                       # csv 直接给了验证集
        train_idx = [i for i, c, _ in matched if c == "train"]
        note = "csv 的 val 行做验证集，train 行做训练集（test 行不参与）"
    else:                             # 没给 val 行：只用 train 子集划分
        pool = [(i, f) for i, c, f in matched if c == "train"]
        if len(pool) < 2:
            raise RuntimeError("csv 里 category=train 的图不足 2 张，无法划分（%s）" % csv_path)
        if any(f is not None for _, f in pool):   # 有 fold 列：fold==0 做验证集
            val_idx = [i for i, f in pool if f == 0]
            train_idx = [i for i, f in pool if f != 0]
            note = "csv 的 fold 列：fold==0 做验证集"
        else:
            idx = [i for i, _ in pool]
            random.Random(seed).shuffle(idx)
            n_val = min(len(idx) - 1, max(1, int(round(len(idx) * val_ratio))))
            val_idx, train_idx = sorted(idx[:n_val]), sorted(idx[n_val:])
            note = "csv 的 train 子集按 seed=%d 随机划分验证集（无 val 行/fold 列）" % seed
    if not val_idx or not train_idx:
        raise RuntimeError("csv 划分后训练/验证集为空，请检查 %s" % csv_path)
    return sorted(train_idx), sorted(val_idx), note


def build_datasets(cfg):
    """按 cfg 构建 (train_ds, val_ds, info)。划分由 seed 决定，同一数据集上不同模型的验证集完全相同。"""
    root = abspath(cfg["data_root"]) / cfg["dataset"]
    pairs = collect_pairs(root / cfg["images_subdir"], root / cfg["labels_subdir"])
    info = scan_label_info(pairs,
                           merge_json_labels=cfg["json_merge_labels"],
                           mask_mode=cfg.get("mask_mode", "auto"),
                           class_names=cfg.get("class_names") or None,
                           root=root, use_cache=bool(cfg.get("label_cache", True)))
    if len(pairs) < 2:
        raise RuntimeError("至少需要 2 张图像才能划分训练/验证集")

    csv_path = find_split_csv(root, cfg.get("split_csv", "auto"))
    split_note = None
    if csv_path is not None:
        train_idx, val_idx, split_note = split_by_csv(pairs, csv_path, cfg["val_ratio"], cfg["seed"])
        print("[data] %s" % split_note)
    else:
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
    info = dict(info, val_files=[pairs[i][0].name for i in val_idx],
                split=split_note or ("random(seed=%d)" % cfg["seed"]),
                n_train=len(train_idx), n_val=len(val_idx))
    return train_ds, val_ds, info
