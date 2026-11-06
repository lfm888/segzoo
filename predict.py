"""预测 / 最终评估 / ONNX 验证（合并了原 predict_onnx.py 的功能）。

三种推理后端，**共用同一份预处理**（inference/preprocess.py，与 C++ 逐行对应）：
  torch      PyTorch 权重（best.pth）
  onnx       onnxruntime（model.onnx）
  dnn        OpenCV DNN（cv::dnn::readNetFromONNX，与 C++ 部署同一套算子）

`--input_dir` 支持三种情况（都能直接跑）：
  1) 留空            -> 用 ckpt 里记录的训练数据集，默认只跑它的 split.txt 的 test 段
  2) 同格式数据集目录 -> 形如 <dir>/train_dataset + <dir>/labels [+ <dir>/split.txt]
                        有 split.txt 就按 --split 取子集（默认 test）；有 labels 就算指标
  3) 纯图像目录       -> 目录里只有图（没 split.txt、没 labels）
                        则对全部图像推理；没有 GT 就不画虚线、不计算指标

输出：每张图一张 <stem>_result.png（原图 + 半透明分割色块 [+ 同色虚线边界]），
      有 GT 时额外输出 <out_dir>/metrics_<split>.json 并打印指标表。

示例：
  python predict.py                                    # test 段，PyTorch
  python predict.py --backend onnx                     # 同目录 model.onnx，onnxruntime
  python predict.py --backend dnn                      # 同目录 model.onnx，OpenCV DNN
  python predict.py --onnx out/model.onnx --backend dnn   # 只有 ONNX + model_meta.json 也能跑
  python predict.py --backend dnn --verbose --timing   # 打印 blob/输出形状与耗时
  python predict.py --input_dir data/new_dataset       # 纯图像目录，全部推理
"""
import json
import time

import numpy as np
import torch

from dataset import (SPLIT_FILE, collect_pairs, mask_to_index, rasterize_labelme, read_mask,
                     read_split_txt, scan_label_info)
from inference.preprocess import (PreprocessSpec, argmax_mask, preprocess, read_image,
                                  resize_mask_nearest, to_model_color)
from models import load_checkpoint, model_desc
from utils import (IMG_EXTS, Progress, abspath, imwrite_unicode, parse_args,
                   print_model_complexity, render_prediction, resize_image, resolve_path, seg_metrics)

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "ckpt": "latest",        # latest = runs 下最新的 best.pth
    "runs_dir": "runs",
    # ---- 推理后端 ----
    "backend": "dnn",      # torch / onnx(onnxruntime) / dnn(OpenCV DNN)
    "onnx": "",              # 空 = 与 ckpt 同目录的 model.onnx；也可直接给 .onnx 路径
                             #      （同目录有 model_meta.json 时不需要 ckpt，用来单独验证导出的 ONNX）
    # ---- 输入 ----
    "input_dir": "",         # 空=训练时的数据集；也可以是【同格式数据集目录】或【纯图像目录】
    "images_subdir": "",     # 指定 input_dir 下的图像子目录（空=自动找 train_dataset/images，否则用 input_dir 本身）
    "labels_subdir": "labels",   # 数据集目录里的标签子目录名
    "split": "test",         # test / val / train / all；该目录没有 split.txt 时自动按 all
    "max_images": 0,         # 0 = 全部
    # ---- 输出 ----
    "out_dir": "",           # 空 = <ckpt 所在目录>/predict（指定了 input_dir 时是 predict_<目录名>）
    "device": "auto",        # auto / cuda / cpu（onnx / dnn 后端固定用 CPU）
    # ---- 诊断 ----
    "verbose": False,        # 打印第一张的 blob / logits 形状与数值范围（验证 ONNX 时很有用）
    "timing": True,          # 统计 preprocess / 推理 / 后处理 的平均耗时
    "progress_bar": True,    # 逐张推理时显示进度条（非终端环境自动改为每 20% 一行）
    "complexity": True,      # 打印模型参数量 / MACs / FLOPs（仅 torch 后端能做，onnx/dnn 见下方提示）
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

# 后端别名 -> 规范名
BACKEND_ALIASES = {
    "torch": "torch", "pt": "torch", "pth": "torch",
    "onnx": "onnxruntime", "onnxruntime": "onnxruntime", "ort": "onnxruntime",
    "dnn": "opencv", "opencv": "opencv", "cv2": "opencv", "opencvdnn": "opencv",
}


# --------------------------------------------------------------------------
# 推理后端：三个后端都只吃 inference.preprocess 产出的 1xCxHxW float32 blob
# --------------------------------------------------------------------------
class TorchRunner:
    name = "torch"

    def __init__(self, model, device):
        self.model, self.device = model, device

    def __call__(self, blob):
        with torch.no_grad():
            return self.model(torch.from_numpy(blob).to(self.device)).cpu().numpy()


class OrtRunner:
    """onnxruntime。输入名从模型里读，不假设叫 "input"。"""
    name = "onnxruntime"

    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, blob):
        return self.sess.run(None, {self.input_name: blob})[0]


class DnnRunner:
    """OpenCV DNN。与 C++ 部署走同一个 cv::dnn 实现，用来验证 ONNX 在部署侧是否可用。"""
    name = "opencv"

    def __init__(self, onnx_path):
        import cv2
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))

    def __call__(self, blob):
        self.net.setInput(blob)
        return self.net.forward()


def normalize_backend(value):
    key = str(value).strip().lower()
    if key not in BACKEND_ALIASES:
        raise SystemExit("backend=%r 不支持，可选: torch / onnx(onnxruntime) / dnn(opencv)"
                         % (value,))
    return BACKEND_ALIASES[key]


def build_runner(backend, onnx_path, model, device):
    if backend == "torch":
        return TorchRunner(model, device)
    if backend == "onnxruntime":
        return OrtRunner(onnx_path)
    return DnnRunner(onnx_path)


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

    roots = []
    for r in (given, images.parent):
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
    root = labels.parent if labels is not None else (split.parent if split is not None else given)
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


def load_meta_and_model(cfg, backend, device):
    """决定 meta / torch 模型 / ONNX 路径。

    - 给了 --onnx 且同目录有 model_meta.json：只用 ONNX，**不需要 ckpt**（单独验证导出的模型）
    - 否则从 best.pth 读 meta；torch 后端还需要模型本体
    返回 (meta, meta_src, ckpt, model, onnx_path)
    """
    onnx_path, meta, meta_src, ckpt, model = None, None, None, None, None

    if cfg["onnx"]:
        onnx_path = abspath(cfg["onnx"])
        if not onnx_path.is_file():
            raise SystemExit("--onnx 指定的文件不存在: %s" % onnx_path)
        meta_file = onnx_path.with_name("model_meta.json")
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta_src = "model_meta.json: %s" % meta_file
        else:
            print("[warn] 没找到 %s，改为从 checkpoint 读取预处理规格" % meta_file.name)

    if meta is None or backend == "torch":
        ckpt = resolve_path(cfg["ckpt"], cfg["runs_dir"], "best.pth")
        model, meta = load_checkpoint(ckpt, device)
        meta_src = "checkpoint: %s" % ckpt

    if backend != "torch":
        if cfg["onnx"] and onnx_path is not None:
            pass
        else:
            base = ckpt.parent if ckpt is not None else abspath(cfg["runs_dir"])
            onnx_path = base / "model.onnx"
        if not onnx_path.is_file():
            raise SystemExit("找不到 ONNX 模型: %s（先跑 python export_onnx.py 导出）" % onnx_path)
        # ONNX 比 checkpoint 旧时，onnx/dnn 会在用户不知情的情况下用旧模型，指标与 torch 对不上
        if ckpt is not None and onnx_path.stat().st_mtime < ckpt.stat().st_mtime:
            print("[warn] %s 比 %s 旧，可能是训练后忘了重新导出；"
                  "此时 backend=onnx/dnn 用的是旧模型（先跑 python export_onnx.py）"
                  % (onnx_path.name, ckpt.name))
    elif cfg["onnx"]:
        print("[warn] backend=torch 不使用 --onnx，已忽略")

    return meta, meta_src, ckpt, model, onnx_path


def main():
    cfg = parse_args(DEFAULTS, "预测 / 最终评估 / ONNX 验证（torch | onnxruntime | opencv dnn）")
    backend = normalize_backend(cfg["backend"])
    use_cuda = torch.cuda.is_available() and cfg["device"] != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")

    meta, meta_src, ckpt, model, onnx_path = load_meta_and_model(cfg, backend, device)
    spec = PreprocessSpec.from_meta(meta)        # 预处理规格：与训练 / ONNX / C++ 同一份契约
    hw = tuple(spec.img_size)
    C = int(meta["num_classes"])
    class_names = list(meta.get("class_names") or [])
    class_names += ["class_%d" % i for i in range(len(class_names), C)]
    runner = build_runner(backend, onnx_path, model, device)

    run_label = ckpt.parent.name if ckpt is not None else onnx_path.parent.name
    model_file = str(onnx_path) if onnx_path is not None else str(ckpt)

    # ---------- 输入目录 & 文件筛选 ----------
    src = resolve_input(cfg, meta)
    files, files_desc, used_split = select_files(list_images(src["images"]), cfg["split"], src["split"])
    if cfg["max_images"] > 0:
        files = files[: cfg["max_images"]]
        files_desc += "（只取前 %d 张）" % cfg["max_images"]
    if cfg["out_dir"]:
        out_dir = abspath(cfg["out_dir"])
    else:
        base = ckpt.parent if ckpt is not None else onnx_path.parent
        out_dir = base / (("predict_" + abspath(cfg["input_dir"]).name) if cfg["input_dir"] else "predict")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[predict] %s | backend=%s | 模型: %s" % (run_label, runner.name, model_file))
    print("[predict] 预处理规格来源: %s" % meta_src)
    print("[predict] 输入 %s" % src["desc"])
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

    # ---------- 预热（避免第一张图的耗时被初始化拉高，影响 timing 统计）----------
    if cfg["timing"] and files:
        # 单通道要传二维数组，三维会被当成 BGR；三通道传 (H,W,3)
        warm_shape = (hw[0], hw[1]) if spec.in_channels == 1 else (hw[0], hw[1], 3)
        warm = preprocess(np.zeros(warm_shape, np.uint8), spec)
        t_w = time.perf_counter()
        runner(warm)
        print("[predict] 预热完成 %.1f ms（不计入统计）" % ((time.perf_counter() - t_w) * 1000))

    # ---------- 模型复杂度（当前推理平台上的模型情况，放在逐张推理之前）----------
    if cfg["complexity"]:
        if model is not None:
            print("[predict] 模型复杂度按 PyTorch 网络统计（backend=%s 只是执行引擎，结构与参数量相同）"
                  % runner.name)
            print_model_complexity(model, spec.in_channels, hw, device, verbose=bool(cfg["verbose"]))
            print("[predict] 开始逐张推理（backend=%s）" % runner.name)
        else:
            print("[predict] backend=%s 没有 torch 模型，跳过复杂度统计"
                  "（ONNX/DNN 的计算量可在 torch 后端下查看）" % runner.name)

    # ---------- 逐张预测 ----------
    conf = np.zeros((C, C), dtype=np.int64) if (info is not None and label_of) else None
    n_scored = 0
    edge_mode = str(cfg["edge"]).lower()
    t_pre = t_inf = t_post = 0.0
    prog = Progress(len(files), "predict", enabled=bool(cfg["progress_bar"]))
    for k, p in enumerate(files, 1):
        # C++ 里的等价流程：cv::imread -> cvtColor -> resize -> (x-mean)/std -> blobFromImage
        t0 = time.perf_counter()
        raw_bgr = read_image(p, spec.in_channels)            # 步骤 1：原分辨率 BGR / 灰度
        x = preprocess(raw_bgr, spec)                        # 步骤 1~5：1xCxHxW float32
        raw = to_model_color(raw_bgr, spec.in_channels)      # 供可视化（RGB / 灰度，仍是原分辨率）
        t_pre += time.perf_counter() - t0

        t0 = time.perf_counter()
        logits = runner(x)                                   # 三个后端统一接口
        t_inf += time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = argmax_mask(logits)                           # 模型输入尺寸上的预测（argmax 取第一个最大值）

        gt_hw, gt_full = None, None
        if info is not None and p.name in label_of:
            gt_hw = np.clip(gt_index_mask(label_of[p.name], info, hw), 0, C - 1)
            conf += np.bincount((gt_hw.astype(np.int64) * C + pred).ravel(), minlength=C * C).reshape(C, C)
            n_scored += 1

        if cfg["save_vis"]:
            # 与训练可视化一致：都画在原图分辨率上（最近邻放大，不产生新类别）
            full = resize_mask_nearest(pred, raw.shape[:2])
            if gt_hw is not None:
                gt_full = resize_mask_nearest(gt_hw, raw.shape[:2])
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
        t_post += time.perf_counter() - t0

        if cfg["verbose"] and k == 1:      # 只打第一张，避免刷屏
            print("[verbose] blob    shape=%s dtype=%s range=[%.3f, %.3f]" % (
                x.shape, x.dtype, float(x.min()), float(x.max())))
            print("[verbose] logits  shape=%s dtype=%s range=[%.3f, %.3f]" % (
                np.shape(logits), np.asarray(logits).dtype,
                float(np.min(logits)), float(np.max(logits))))
            if tuple(np.shape(logits))[-2:] != hw:
                print("[warn] 输出空间尺寸 %s 与输入 %s 不一致，请检查模型" % (
                    tuple(np.shape(logits))[-2:], hw))

        prog.update(k, p.name if gt_hw is not None or info is None else "%s (无标签)" % p.name)
    prog.close()

    # ---------- 耗时统计 ----------
    if cfg["timing"] and files:
        n = float(len(files))
        print("\n[timing] 平均每张：预处理 %.2f ms + 推理 %.2f ms + 后处理 %.2f ms = %.2f ms"
              "（backend=%s, 输入 %dx%d）" % (
                  t_pre / n * 1000, t_inf / n * 1000, t_post / n * 1000,
                  (t_pre + t_inf + t_post) / n * 1000, runner.name, hw[0], hw[1]))

    # ---------- 指标汇总 ----------
    if conf is not None and n_scored:
        m = seg_metrics(conf, C)
        print("\n" + "=" * 78)
        print("最终评估（%s 段）| %s | %s @ %dx%d | backend=%s | 统计 %d 张" % (
            used_split, run_label, model_desc(meta["arch"], meta["encoder"]), hw[0], hw[1],
            runner.name, n_scored))
        print("-" * 78)
        print("  %-4s %-14s %-8s %-8s %-8s %-8s %-8s" % ("cls", "name", "Dice", "IoU", "P", "R", "F1"))
        for c in range(C):
            print("  %-4d %-14s %-8s %-8s %-8s %-8s %-8s" % (
                c, str(class_names[c])[:14],
                *["%.4f" % v if v == v else "-" for v in (m["dice"][c], m["iou"][c], m["prec"][c],
                                                          m["rec"][c], m["f1"][c])]))
        print("-" * 78)
        print("  前景均值    Dice %.4f  IoU %.4f  P %.4f  R %.4f  F1 %.4f" % (
            m["mdice"], m["miou"], m["mprec"], m["mrec"], m["mf1"]))
        print("  准确率      前景像素 %.4f   全部像素 %.4f   （前景占 %.2f%%）" % (
            m["acc_fg"], m["pacc"], 100.0 * m["fg_ratio"]))
        print("=" * 78)
        res = {"split": used_split, "n_images": n_scored, "backend": runner.name,
               "model": model_file, "ckpt": str(ckpt) if ckpt is not None else None,
               "input_dir": str(src["images"]),
               "arch": meta["arch"], "encoder": meta["encoder"], "img_size": [hw[0], hw[1]],
               "num_classes": C, "class_names": class_names,
               "mdice": m["mdice"], "miou": m["miou"], "mprec": m["mprec"], "mrec": m["mrec"],
               "mf1": m["mf1"], "pacc": m["pacc"], "acc_fg": m["acc_fg"], "fg_ratio": m["fg_ratio"],
               "dice": m["dice"], "iou": m["iou"], "prec": m["prec"], "rec": m["rec"], "f1": m["f1"]}
        if cfg["timing"]:
            n = float(len(files))
            res["latency_ms_per_image"] = round((t_pre + t_inf + t_post) / n * 1000, 3)
        out_json = out_dir / ("metrics_%s.json" % used_split)
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[predict] 指标已保存: %s" % out_json)
    elif cfg["metrics"] and info is not None:
        print("[predict] 没有可统计的标签，跳过指标")

    print("[predict] 完成%s" % ("（每张 *_result.png）" if cfg["save_vis"] else ""))


if __name__ == "__main__":
    main()
