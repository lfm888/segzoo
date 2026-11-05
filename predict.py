"""用训练好的模型做预测 / 最终评估，支持任意图像目录。

`--input_dir` 支持三种情况（都能直接跑）：
  1) 留空            -> 用 ckpt 里记录的训练数据集，默认只跑它的 split.txt 的 test 段
  2) 同格式数据集目录 -> 形如 <dir>/train_dataset + <dir>/labels [+ <dir>/split.txt]
                        有 split.txt 就按 --split 取子集（默认 test）；有 labels 就算指标
  3) 纯图像目录       -> 目录里只有图（没 split.txt、没 labels）
                        则对全部图像推理；没有 GT 就不画虚线、不计算指标

输出：每张图一张 <stem>_result.png（原图 + 半透明分割色块 [+ 同色虚线边界]），
      有 GT 时额外输出 <out_dir>/metrics_<split>.json 并打印指标表。

示例：
  python predict.py                                   # 训练数据集的 test 段
  python predict.py --input_dir data/new_dataset      # 纯图像目录，全部推理
  python predict.py --input_dir data/OCT5K --split test
"""
import json

import cv2
import numpy as np
import torch

from dataset import (SPLIT_FILE, collect_pairs, mask_to_index, rasterize_labelme, read_mask,
                     read_split_txt, scan_label_info)
from models import load_checkpoint
from utils import (IMG_EXTS, abspath, imwrite_unicode, load_image, normalize, parse_args,
                   render_prediction, resize_image, resolve_path, seg_metrics)

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "ckpt": "latest",        # latest = runs 下最新的 best.pth
    "runs_dir": "runs",
    "backend": "torch",      # torch / onnx（onnx 用同目录 model.onnx，用来验证导出后的模型）
    # ---- 输入 ----
    "input_dir": "",         # 空=训练时的数据集；也可以是【同格式数据集目录】或【纯图像目录】
    "images_subdir": "",     # 指定 input_dir 下的图像子目录（空=自动找 train_dataset/images，否则用 input_dir 本身）
    "labels_subdir": "labels",   # 数据集目录里的标签子目录名
    "split": "test",         # test / val / train / all；该目录没有 split.txt 时自动按 all
    "max_images": 0,         # 0 = 全部
    # ---- 输出 ----
    "out_dir": "",           # 空 = <ckpt 所在目录>/predict（指定了 input_dir 时是 predict_<目录名>）
    "device": "auto",        # auto / cuda / cpu（onnx 后端固定用 CPU）
    # ---- 最终评估 ----
    "metrics": True,         # 有标签就算指标（Dice/IoU/P/R/F1），存 <out_dir>/metrics_<split>.json
    "save_vis": True,        # 是否保存 *_result.png（只要指标时可设 False 提速）
    # ---- 可视化 ----
    "edge": "auto",          # 虚线画谁的轮廓：auto(有GT用GT，没GT就不画) / gt / pred / none
    "alpha": 0.45,           # 预测区域的填充透明度（0=只看虚线，1=完全不透明）
    "dash": 0,               # 虚线实线段长度（像素），0 = 按线宽自动
    "gap": 0,                # 虚线间隔长度（像素），0 = 按线宽自动
    "thickness": 0,          # 虚线线宽，0 = 按图像尺寸自动
    "legend": True,          # 左上角是否画类别图例
}
# ===========================================================================

IMAGE_SUBDIR_CANDIDATES = ("train_dataset", "images", "imgs", "JPEGImages")


def list_images(d):
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in IMG_EXTS and not p.stem.endswith(("_result", "_mask", "_panel")))


def resolve_input(cfg, meta):
    """把 --input_dir 解析成 {images, root, labels, split, desc}：
       images: 图像目录 | root: 数据集根（放 split.txt / labels） | labels: 标签目录或 None
       split : split.txt 路径或 None"""
    ck_root = abspath(meta["data_root"]) / meta["dataset"]
    if not cfg["input_dir"]:      # 1) 默认：训练时用的数据集
        return {"images": ck_root / meta["images_subdir"], "root": ck_root,
                "labels": ck_root / meta.get("labels_subdir", "labels"),
                "split": ck_root / SPLIT_FILE, "desc": "训练数据集 %s" % meta["dataset"]}

    given = abspath(cfg["input_dir"])
    if not given.is_dir():
        raise SystemExit("input_dir 不存在: %s" % given)

    # 找图像目录：显式指定 > 常见子目录 > 目录本身就是图像目录
    images = None
    if cfg["images_subdir"]:
        images = given / cfg["images_subdir"]
    else:
        for sub in IMAGE_SUBDIR_CANDIDATES + (meta["images_subdir"],):
            if (given / sub).is_dir() and list_images(given / sub):
                images = given / sub
                break
    if images is None:
        images = given                      # 3) 纯图像目录
    if not list_images(images):
        raise SystemExit("在 %s 下没找到图像（支持 %s）" % (images, ", ".join(sorted(IMG_EXTS))))

    root = given if images == given or images.parent == given else images.parent
    roots = []
    for r in (root, images.parent):
        if r not in roots:
            roots.append(r)

    labels = None
    for r in roots:
        for name in (cfg["labels_subdir"], meta.get("labels_subdir", "labels")):
            if (r / name).is_dir():
                labels = r / name
                break
        if labels is not None:
            break
    split = next((r / SPLIT_FILE for r in roots if (r / SPLIT_FILE).is_file()), None)
    if labels is not None:
        root = labels.parent
    elif split is not None:
        root = split.parent
    desc = "%s（图像: %s%s%s）" % (
        given.name, images.name if images != given else "本目录",
        "，标签: 有" if labels else "，标签: 无", "，split.txt: 有" if split else "，split.txt: 无")
    return {"images": images, "root": root, "labels": labels, "split": split, "desc": desc}


def select_files(files, split, split_path):
    """按 split.txt 选子集。返回 (文件列表, 说明, 实际使用的分段名)；
    没有 split.txt 或 split=all 时用全部图像，分段名记为 all。"""
    if split in (None, "", "all"):
        return files, "全部 %d 张" % len(files), "all"
    table = read_split_txt(split_path) if split_path and split_path.is_file() else None
    if not table:
        print("[warn] 没找到 %s，按全部图像推理（split=%s 忽略）" % (SPLIT_FILE, split))
        return files, "全部 %d 张（无 split.txt）" % len(files), "all"
    want = {n for n, s in table.items() if s == split}
    picked = [p for p in files if p.name in want]
    if not picked:
        raise SystemExit("split.txt（%s）里没有 %s 段的图像，可选段: %s" % (
            split_path, split, ", ".join(sorted({v for v in table.values()}))))
    return picked, "%s.txt 的 %s 段 %d 张" % (split_path.stem, split, len(picked)), split


def gt_index_mask(lp, info, hw):
    """按标签文件取类别索引图，并缩放到模型输入尺寸（最近邻），用于算指标。"""
    if info["kind"] == "json":
        m = rasterize_labelme(lp, info["label2idx"], hw)
    else:
        m = mask_to_index(read_mask(lp), info)
    return resize_image(m, hw, is_mask=True)


def main():
    cfg = parse_args(DEFAULTS, "批量预测 / 在 split 的 test 段做最终评估")
    ckpt = resolve_path(cfg["ckpt"], cfg["runs_dir"], "best.pth")
    use_cuda = torch.cuda.is_available() and cfg["device"] != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    model, meta = load_checkpoint(ckpt, device)
    hw = tuple(meta["img_size"])
    C = int(meta["num_classes"])
    class_names = list(meta.get("class_names") or [])
    class_names += ["class_%d" % i for i in range(len(class_names), C)]

    sess = None
    if cfg["backend"] == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ckpt.parent / "model.onnx"), providers=["CPUExecutionProvider"])

    # ---------- 输入目录 & 文件筛选 ----------
    src = resolve_input(cfg, meta)
    files, files_desc, used_split = select_files(list_images(src["images"]), cfg["split"], src["split"])
    if cfg["max_images"] > 0:
        files = files[: cfg["max_images"]]
        files_desc += "（只取前 %d 张）" % cfg["max_images"]
    out_dir = (abspath(cfg["out_dir"]) if cfg["out_dir"]
               else ckpt.parent / (("predict_" + abspath(cfg["input_dir"]).name) if cfg["input_dir"] else "predict"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[predict] %s | backend=%s | 输入 %s" % (ckpt.parent.name, cfg["backend"], src["desc"]))
    print("[predict] 图像目录 %s | %s | %d 张 -> %s" % (src["images"], files_desc, len(files), out_dir))

    # ---------- GT（有才算指标 / 画虚线）----------
    label_of, info = {}, None
    if src["labels"] is not None:
        try:
            pairs = collect_pairs(src["images"], src["labels"])
            info = scan_label_info(pairs, merge_json_labels=bool(meta.get("json_merge_labels", False)),
                                   mask_mode="auto", class_names=meta.get("class_names") or None,
                                   root=src["root"], use_cache=True)
            if int(info["num_classes"]) != C:
                print("[warn] 标签类别数 %d 与模型 %d 不一致，按模型类别数截断/补齐计算" % (
                    info["num_classes"], C))
            label_of = {ip.name: lp for ip, lp in pairs}
            miss = [p.name for p in files if p.name not in label_of]
            if miss:
                print("[warn] %d 张图没有对应标签，不参与指标（示例: %s）" % (len(miss), miss[:3]))
        except Exception as e:
            print("[warn] 读不到标签，只出可视化不算指标: %s: %s" % (type(e).__name__, e))
            info, label_of = None, {}
    else:
        print("[predict] 没有 labels，只做推理（不画 GT 虚线、不算指标）")

    # ---------- 逐张预测 ----------
    conf = np.zeros((C, C), dtype=np.int64) if (info is not None and label_of) else None
    n_scored = 0
    edge_mode = str(cfg["edge"]).lower()
    for k, p in enumerate(files, 1):
        raw = load_image(p, meta["in_channels"])
        x = normalize(resize_image(raw, hw), meta["mean"], meta["std"])[None]
        if sess is not None:
            logits = sess.run(None, {"input": x})[0]
        else:
            with torch.no_grad():
                logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
        pred = logits[0].argmax(0).astype(np.uint8)          # 模型输入尺寸上的预测

        gt_hw, gt_full = None, None
        if info is not None and p.name in label_of:
            gt_hw = np.clip(gt_index_mask(label_of[p.name], info, hw), 0, C - 1)
            conf += np.bincount((gt_hw.astype(np.int64) * C + pred).ravel(), minlength=C * C).reshape(C, C)
            n_scored += 1

        if cfg["save_vis"]:
            # 与训练可视化一致：都画在原图分辨率上（最近邻放大，不产生新类别）
            full = cv2.resize(pred, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
            if gt_hw is not None:
                gt_full = cv2.resize(gt_hw, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
            if edge_mode == "pred":
                edge_mask, draw = full, True
            elif edge_mode in ("none", "off", "false") or (edge_mode in ("auto", "gt") and gt_full is None):
                edge_mask, draw = None, False
            else:                                  # gt（有 GT 时）
                edge_mask, draw = gt_full, True
            vis = render_prediction(raw, full, alpha=cfg["alpha"], class_names=class_names,
                                    dash=cfg["dash"], gap=cfg["gap"],
                                    thickness=cfg["thickness"] or None, legend=cfg["legend"],
                                    edge_mask=edge_mask, draw_edges=draw)
            imwrite_unicode(out_dir / (p.stem + "_result.png"), vis)
        print("  [%d/%d] %s%s" % (k, len(files), p.name,
                                 "" if gt_hw is not None else ("  (无标签)" if info else "")))

    # ---------- 指标汇总 ----------
    if conf is not None and n_scored:
        m = seg_metrics(conf, C)
        print("\n" + "=" * 78)
        print("最终评估（%s 段）| %s | %s + %s @ %dx%d | 统计 %d 张" % (
            used_split, ckpt.parent.name, meta["arch"], meta["encoder"], hw[0], hw[1], n_scored))
        print("-" * 78)
        print("  %-4s %-14s %-8s %-8s %-8s %-8s %-8s" % ("cls", "name", "Dice", "IoU", "P", "R", "F1"))
        for c in range(C):
            print("  %-4d %-14s %-8s %-8s %-8s %-8s %-8s" % (
                c, str(class_names[c])[:14],
                *["%.4f" % v if v == v else "-" for v in (m["dice"][c], m["iou"][c], m["prec"][c],
                                                          m["rec"][c], m["f1"][c])]))
        print("-" * 78)
        print("  前景均值    Dice %.4f  IoU %.4f  P %.4f  R %.4f  F1 %.4f   像素准确率 %.4f" % (
            m["mdice"], m["miou"], m["mprec"], m["mrec"], m["mf1"], m["pacc"]))
        print("=" * 78)
        res = {"split": used_split, "n_images": n_scored, "ckpt": str(ckpt),
               "input_dir": str(src["images"]),
               "arch": meta["arch"], "encoder": meta["encoder"], "img_size": [hw[0], hw[1]],
               "num_classes": C, "class_names": class_names,
               "mdice": m["mdice"], "miou": m["miou"], "mprec": m["mprec"], "mrec": m["mrec"],
               "mf1": m["mf1"], "pacc": m["pacc"],
               "dice": m["dice"], "iou": m["iou"], "prec": m["prec"], "rec": m["rec"], "f1": m["f1"]}
        out_json = out_dir / ("metrics_%s.json" % used_split)
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[predict] 指标已保存: %s" % out_json)
    elif cfg["metrics"] and info is not None:
        print("[predict] 没有可统计的标签，跳过指标")

    print("[predict] 完成%s" % ("（每张 *_result.png）" if cfg["save_vis"] else ""))


if __name__ == "__main__":
    main()
