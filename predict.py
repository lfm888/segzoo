"""用训练好的模型对一个文件夹的图像做预测（不需要标签，适合在其它设备拍摄的新图上看泛化）。

直接运行：python predict.py
输出：<out_dir>/xxx_result.png —— 只存一张图，同时表达 原图 / mask 边界 / 分割结果：
  · 底图 = 原图（原分辨率）
  · 每个类别 = 同色「虚线边界」+ 同色「半透明填充」（填充颜色与虚线颜色一致）
  · 左上角图例列出当前图里出现的类别（带类名，来自训练时的 meta）
"""
import cv2
import numpy as np
import torch

from models import load_checkpoint
from utils import (IMG_EXTS, abspath, imwrite_unicode, load_image, normalize, parse_args,
                   render_prediction, resolve_path, resize_image)

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "ckpt": "latest",        # latest = runs 下最新的 best.pth
    "runs_dir": "runs",
    "backend": "torch",      # torch / onnx（onnx 用同目录 model.onnx，用来验证导出后的模型）
    "input_dir": "",         # 空 = 训练时的 <data_root>/<dataset>/train_dataset
    "out_dir": "",           # 空 = <ckpt 所在目录>/predict
    "max_images": 20,        # 0 = 全部
    "device": "auto",        # auto / cuda / cpu（onnx 后端固定用 CPU）
    # ---- 可视化 ----
    "alpha": 0.45,           # 预测区域的填充透明度（0=只看虚线，1=完全不透明）
    "dash": 0,               # 虚线实线段长度（像素），0 = 按线宽自动
    "gap": 0,                # 虚线间隔长度（像素），0 = 按线宽自动
    "thickness": 0,          # 虚线线宽，0 = 按图像尺寸自动
    "legend": True,          # 左上角是否画类别图例
}
# ===========================================================================


def main():
    cfg = parse_args(DEFAULTS, "批量预测并保存单张可视化结果")
    ckpt = resolve_path(cfg["ckpt"], cfg["runs_dir"], "best.pth")
    use_cuda = torch.cuda.is_available() and cfg["device"] != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    model, meta = load_checkpoint(ckpt, device)
    C, hw = meta["in_channels"], tuple(meta["img_size"])

    sess = None
    if cfg["backend"] == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ckpt.parent / "model.onnx"), providers=["CPUExecutionProvider"])

    in_dir = abspath(cfg["input_dir"]) if cfg["input_dir"] else abspath(meta["data_root"]) / meta["dataset"] / meta["images_subdir"]
    out_dir = abspath(cfg["out_dir"]) if cfg["out_dir"] else ckpt.parent / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS and not p.stem.endswith(("_result", "_mask", "_panel")))
    if cfg["max_images"] > 0:
        files = files[: cfg["max_images"]]
    print("[predict] %s | backend=%s | %d 张 | %s -> %s" % (
        ckpt.parent.name, cfg["backend"], len(files), in_dir, out_dir))

    for p in files:
        raw = load_image(p, C)                      # 原分辨率图（灰度 HxW 或 RGB HxWx3）
        x = normalize(resize_image(raw, hw), meta["mean"], meta["std"])[None]
        if sess is not None:
            logits = sess.run(None, {"input": x})[0]
        else:
            with torch.no_grad():
                logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
        pred = logits[0].argmax(0).astype(np.uint8)

        # 预测图放大回原图尺寸（最近邻，不产生新的类别值）
        full = cv2.resize(pred, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
        vis = render_prediction(raw, full, alpha=cfg["alpha"], class_names=meta.get("class_names"),
                                dash=cfg["dash"], gap=cfg["gap"],
                                thickness=cfg["thickness"] or None, legend=cfg["legend"])
        imwrite_unicode(out_dir / (p.stem + "_result.png"), vis)
    print("[predict] 完成：每张图只输出一个 *_result.png（原图 + 半透明分割色块 + 同色虚线边界）")


if __name__ == "__main__":
    main()
