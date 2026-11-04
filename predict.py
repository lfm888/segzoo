"""用训练好的模型对一个文件夹的图像做预测（不需要标签，适合在其它设备拍摄的新图上看泛化）。

直接运行：python predict.py
输出：<out_dir>/xxx_panel.png（原图 | 预测叠加 | 预测彩色图）、xxx_mask.png（原图尺寸的类别索引图）
"""
import cv2
import numpy as np
import torch

from models import load_checkpoint
from utils import (IMG_EXTS, abspath, colorize, imwrite_unicode, load_image, normalize, overlay,
                   parse_args, resolve_path, resize_image, to_bgr)

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "ckpt": "latest",        # latest = runs 下最新的 best.pth
    "runs_dir": "runs",
    "backend": "torch",      # torch / onnx（onnx 用同目录 model.onnx，用来验证导出后的模型）
    "input_dir": "",         # 空 = 训练时的 <data_root>/<dataset>/train_dataset
    "out_dir": "",           # 空 = <ckpt 所在目录>/predict
    "max_images": 20,        # 0 = 全部
    "device": "auto",        # auto / cuda / cpu（onnx 后端固定用 CPU）
}
# ===========================================================================


def main():
    cfg = parse_args(DEFAULTS, "批量预测并保存可视化")
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
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if cfg["max_images"] > 0:
        files = files[: cfg["max_images"]]
    print("[predict] %s | backend=%s | %d 张 | %s -> %s" % (ckpt.parent.name, cfg["backend"], len(files), in_dir, out_dir))

    for p in files:
        raw = load_image(p, C)
        img = resize_image(raw, hw)
        x = normalize(img, meta["mean"], meta["std"])[None]
        if sess is not None:
            logits = sess.run(None, {"input": x})[0]
        else:
            with torch.no_grad():
                logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
        pred = logits[0].argmax(0).astype(np.uint8)

        panel = np.hstack([to_bgr(img), overlay(img, pred), colorize(pred)])
        imwrite_unicode(out_dir / (p.stem + "_panel.png"), panel)
        full = cv2.resize(pred, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
        imwrite_unicode(out_dir / (p.stem + "_mask.png"), full)
    print("[predict] 完成")


if __name__ == "__main__":
    main()
