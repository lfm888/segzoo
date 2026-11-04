"""ONNX -> RKNN。只能在装有 rknn-toolkit2 的 Linux（或 WSL2/Docker）里运行，Windows 原生不支持。

直接运行：python convert_rknn.py       （自动取 runs 下最新的 model.onnx 及同目录 model_meta.json）
会用训练集图像生成量化校准集（缩放到模型输入尺寸）。
注意：本脚本按 RKNN-Toolkit2 常见 API 编写，我没有在带 NPU 工具链的环境里实测过，
      如与你安装的版本 API 有出入，按报错微调 config/build 参数即可。
"""
import json
import random

from utils import (IMG_EXTS, abspath, imwrite_unicode, load_image, parse_args, resize_image, resolve_path, to_bgr)

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "onnx": "latest",             # latest = runs 下最新的 model.onnx
    "runs_dir": "runs",
    "target_platform": "rk3568",  # rk3566/rk3568/rk3588 等
    "do_quant": True,             # RK356x 的 NPU 以 INT8 为主，建议量化；False 则不量化
    "quant_images": 100,          # 量化校准图数量
    "quant_seed": 0,
    "out": "",                    # 空 = 与 onnx 同目录的 model.rknn
}
# ===========================================================================


def make_quant_set(meta, onnx_dir, n, seed):
    in_dir = abspath(meta["data_root"]) / meta["dataset"] / meta["images_subdir"]
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    random.Random(seed).shuffle(files)
    qdir = onnx_dir / "quant"
    qdir.mkdir(exist_ok=True)
    lines = []
    for p in files[:n]:
        img = resize_image(load_image(p, meta["in_channels"]), tuple(meta["img_size"]))
        dst = qdir / (p.stem + ".png")
        imwrite_unicode(dst, to_bgr(img) if meta["in_channels"] == 3 else img)  # 3 通道存成 BGR 文件，读回为 RGB 语义由 toolkit 处理
        lines.append(str(dst))
    txt = onnx_dir / "quant_list.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")
    return txt


def main():
    cfg = parse_args(DEFAULTS, "ONNX -> RKNN")
    from rknn.api import RKNN  # 仅在 Linux + rknn-toolkit2 环境可用

    onnx_path = resolve_path(cfg["onnx"], cfg["runs_dir"], "model.onnx")
    meta = json.load(open(onnx_path.parent / "model_meta.json", encoding="utf-8"))
    C = meta["in_channels"]
    out = abspath(cfg["out"]) if cfg["out"] else onnx_path.parent / "model.rknn"

    rknn = RKNN(verbose=False)
    # RKNN 内部做 (x - mean) / std，与训练时的 normalize 完全一致（0~255 单位）
    rknn.config(mean_values=[meta["mean"] if len(meta["mean"]) == C else [meta["mean"][0]] * C],
                std_values=[meta["std"] if len(meta["std"]) == C else [meta["std"][0]] * C],
                target_platform=cfg["target_platform"])
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit("load_onnx 失败")
    dataset = str(make_quant_set(meta, onnx_path.parent, cfg["quant_images"], cfg["quant_seed"])) if cfg["do_quant"] else None
    if rknn.build(do_quantization=cfg["do_quant"], dataset=dataset) != 0:
        raise SystemExit("build 失败")
    if rknn.export_rknn(str(out)) != 0:
        raise SystemExit("export_rknn 失败")
    rknn.release()
    print("[rknn] 已生成 %s（输出为 logits，argmax 在主机端做）" % out)


if __name__ == "__main__":
    main()
