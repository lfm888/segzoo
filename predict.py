"""用训练好的模型做预测，并（默认）在 **test 段** 上做最终泛化评估。

数据划分是固定的：<dataset>/split.txt（train/val/test），train.py 只用 train+val，
test 段保留到这里评估。有标签时直接算 Dice/IoU/P/R/F1 并存 metrics_<split>.json。

直接运行：python predict.py
输出：每张图一张 xxx_result.png（原图 + 半透明分割色块 + 同色虚线边界 + 图例），
      默认只跑 test 段；有标签时额外输出 <out_dir>/metrics_test.json 并打印指标表。
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
    "input_dir": "",         # 空 = 训练时的 <data_root>/<dataset>/train_dataset
    "out_dir": "",           # 空 = <ckpt 所在目录>/predict
    "split": "test",         # 用哪一段评估：test / val / train / all（按 <dataset>/split.txt）
    "max_images": 0,         # 0 = 全部（评估时建议 0）
    "device": "auto",        # auto / cuda / cpu（onnx 后端固定用 CPU）
    # ---- 最终评估 ----
    "metrics": True,         # 有标签就算指标（Dice/IoU/P/R/F1），存 <out_dir>/metrics_<split>.json
    "save_vis": True,        # 是否保存 *_result.png（只要指标时可设 False 提速）
    # ---- 可视化 ----
    "alpha": 0.45,           # 预测区域的填充透明度（0=只看虚线，1=完全不透明）
    "dash": 0,               # 虚线实线段长度（像素），0 = 按线宽自动
    "gap": 0,                # 虚线间隔长度（像素），0 = 按线宽自动
    "thickness": 0,          # 虚线线宽，0 = 按图像尺寸自动
    "legend": True,          # 左上角是否画类别图例
}
# ===========================================================================


def pick_files(in_dir, split, split_path):
    """按 split.txt 选要评估的文件名集合；返回 (文件名集合或 None=全部, 说明)。"""
    if split in (None, "", "all"):
        return None, "全部文件"
    table = read_split_txt(split_path) if split_path and split_path.is_file() else None
    if not table:
        return None, "未找到 %s，使用全部文件" % (split_path.name if split_path else SPLIT_FILE)
    want = {n for n, s in table.items() if s == split}
    return want, "split.txt 的 %s 段（%d 张）" % (split, len(want))


def gt_index_mask(lp, info, hw):
    """按标签文件取类别索引图，并缩放到模型输入尺寸（最近邻），用于和预测对齐着算指标。"""
    if info["kind"] == "json":
        m = rasterize_labelme(lp, info["label2idx"], hw)     # labelme 多边形栅格化
    else:
        m = mask_to_index(read_mask(lp), info)
    return resize_image(m, hw, is_mask=True)


def main():
    cfg = parse_args(DEFAULTS, "批量预测 / 在 test 段做最终评估")
    ckpt = resolve_path(cfg["ckpt"], cfg["runs_dir"], "best.pth")
    use_cuda = torch.cuda.is_available() and cfg["device"] != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    model, meta = load_checkpoint(ckpt, device)
    hw = tuple(meta["img_size"])
    C = int(meta["num_classes"])

    sess = None
    if cfg["backend"] == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ckpt.parent / "model.onnx"), providers=["CPUExecutionProvider"])

    data_root = abspath(meta["data_root"]) / meta["dataset"]
    in_dir = abspath(cfg["input_dir"]) if cfg["input_dir"] else data_root / meta["images_subdir"]
    out_dir = abspath(cfg["out_dir"]) if cfg["out_dir"] else ckpt.parent / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 选文件（默认只跑 test 段）----------
    split_path = data_root / SPLIT_FILE
    default_in = data_root / meta["images_subdir"]
    if in_dir != default_in and cfg["split"] not in ("", "all"):
        print("[warn] 指定了外部 input_dir（%s），split.txt 只对数据集目录有效 -> 按全部文件处理" % in_dir)
        want, split_desc = None, "全部文件"
    else:
        want, split_desc = pick_files(in_dir, cfg["split"], split_path)
    files = sorted(p for p in in_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS and not p.stem.endswith(("_result", "_mask", "_panel")))
    if want is not None:
        files = [p for p in files if p.name in want]
    if cfg["max_images"] > 0:
        files = files[: cfg["max_images"]]
    if not files:
        raise SystemExit("[predict] 没有可评估的图像（split=%s，目录 %s）" % (cfg["split"], in_dir))
    print("[predict] %s | backend=%s | split=%s -> %s | %d 张 | %s -> %s" % (
        ckpt.parent.name, cfg["backend"], cfg["split"], split_desc, len(files), in_dir, out_dir))

    # ---------- 需要标签才能算指标 ----------
    label_of, info = {}, None
    if cfg["metrics"]:
        try:
            pairs = collect_pairs(in_dir, data_root / meta.get("labels_subdir", "labels"))
            info = scan_label_info(pairs, merge_json_labels=meta.get("json_merge_labels", False),
                                   mask_mode="auto", class_names=meta.get("class_names") or None,
                                   root=data_root, use_cache=True)
            label_of = {ip.name: lp for ip, lp in pairs}
            miss = [p.name for p in files if p.name not in label_of]
            if miss:
                print("[warn] %d 张图没有标签，指标只统计有标签的 %d 张（示例: %s）" % (
                    len(miss), len(files) - len(miss), miss[:3]))
        except Exception as e:
            print("[warn] 读不到标签，跳过指标（只出可视化）: %s: %s" % (type(e).__name__, e))
            info = None

    # ---------- 逐张预测 ----------
    conf = np.zeros((C, C), dtype=np.int64) if info is not None else None
    n_scored = 0
    for k, p in enumerate(files, 1):
        raw = load_image(p, meta["in_channels"])
        x = normalize(resize_image(raw, hw), meta["mean"], meta["std"])[None]
        if sess is not None:
            logits = sess.run(None, {"input": x})[0]
        else:
            with torch.no_grad():
                logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
        pred = logits[0].argmax(0).astype(np.uint8)

        if conf is not None and p.name in label_of:
            gt = gt_index_mask(label_of[p.name], info, hw)
            gt = np.clip(gt, 0, C - 1)
            conf += np.bincount((gt.astype(np.int64) * C + pred).ravel(), minlength=C * C).reshape(C, C)
            n_scored += 1

        if cfg["save_vis"]:
            full = cv2.resize(pred, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
            vis = render_prediction(raw, full, alpha=cfg["alpha"], class_names=meta.get("class_names"),
                                    dash=cfg["dash"], gap=cfg["gap"],
                                    thickness=cfg["thickness"] or None, legend=cfg["legend"])
            imwrite_unicode(out_dir / (p.stem + "_result.png"), vis)
        print("  [%d/%d] %s" % (k, len(files), p.name) + (
            "" if cfg["save_vis"] else "  (只算指标)"))

    # ---------- 指标汇总 ----------
    if conf is not None and n_scored:
        m = seg_metrics(conf, C)
        names = list(meta.get("class_names") or []) + ["class_%d" % i for i in range(len(meta.get("class_names") or []), C)]
        print("\n" + "=" * 78)
        print("最终评估（%s 段）| %s | %s + %s @ %dx%d | 统计 %d 张" % (
            cfg["split"], ckpt.parent.name, meta["arch"], meta["encoder"], hw[0], hw[1], n_scored))
        print("-" * 78)
        print("  %-4s %-14s %-8s %-8s %-8s %-8s %-8s" % ("cls", "name", "Dice", "IoU", "P", "R", "F1"))
        for c in range(C):
            print("  %-4d %-14s %-8s %-8s %-8s %-8s %-8s" % (
                c, str(names[c])[:14],
                *["%.4f" % v if v == v else "-" for v in (m["dice"][c], m["iou"][c], m["prec"][c],
                                                          m["rec"][c], m["f1"][c])]))
        print("-" * 78)
        print("  前景均值    Dice %.4f  IoU %.4f  P %.4f  R %.4f  F1 %.4f   像素准确率 %.4f" % (
            m["mdice"], m["miou"], m["mprec"], m["mrec"], m["mf1"], m["pacc"]))
        print("=" * 78)
        res = {"split": cfg["split"], "n_images": n_scored, "ckpt": str(ckpt),
               "arch": meta["arch"], "encoder": meta["encoder"], "img_size": [hw[0], hw[1]],
               "num_classes": C, "class_names": names,
               "mdice": m["mdice"], "miou": m["miou"], "mprec": m["mprec"], "mrec": m["mrec"],
               "mf1": m["mf1"], "pacc": m["pacc"],
               "dice": m["dice"], "iou": m["iou"], "prec": m["prec"], "rec": m["rec"], "f1": m["f1"]}
        out_json = out_dir / ("metrics_%s.json" % cfg["split"])
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[predict] 指标已保存: %s" % out_json)
    elif cfg["metrics"]:
        print("[predict] 没有可统计的标签，跳过指标")

    print("[predict] 完成%s" % ("（每张 *_result.png：原图 + 半透明色块 + 同色虚线边界）" if cfg["save_vis"] else ""))


if __name__ == "__main__":
    main()
