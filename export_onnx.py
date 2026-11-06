"""把训练好的 best.pth 导出为 ONNX（固定输入尺寸、静态 shape，便于转 RKNN），并用 onnxruntime 校验。

直接运行：python export_onnx.py        （自动取 runs 下最新的 best.pth）
指定模型：python export_onnx.py --ckpt runs/AIDK/unet_resnet18_288x512/best.pth
输出：同目录下 model.onnx 与 model_meta.json（记录输入尺寸/均值方差/类别映射，供 predict、convert_rknn 使用）
"""
import json

import numpy as np
import torch

from models import load_checkpoint
from utils import abspath, parse_args, resolve_path

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "ckpt": "latest",       # latest = runs 下修改时间最新的 best.pth；也可写具体路径
    "runs_dir": "runs",
    "out": "",              # 空 = 与 ckpt 同目录的 model.onnx
    "opset": 12,            # RKNN-Toolkit2 对 opset 12 兼容最稳；失败时可试 13
    "simplify": True,       # 用 onnxsim 化简（未安装会自动跳过）
    "verify": True,         # 与 PyTorch 输出对比
}
# ===========================================================================


def main():
    cfg = parse_args(DEFAULTS, "导出 ONNX")
    ckpt = resolve_path(cfg["ckpt"], cfg["runs_dir"], "best.pth")
    model, meta = load_checkpoint(ckpt, "cpu")
    C, (H, W) = meta["in_channels"], meta["img_size"]
    out = abspath(cfg["out"]) if cfg["out"] else ckpt.parent / "model.onnx"
    print("[export] %s -> %s  输入 (1,%d,%d,%d)" % (ckpt, out, C, H, W))

    dummy = torch.randn(1, C, H, W)
    torch.onnx.export(model, dummy, str(out), opset_version=cfg["opset"], input_names=["input"],
                      output_names=["logits"], do_constant_folding=True)  # 不设 dynamic_axes：RKNN 要静态 shape

    if cfg["simplify"]:
        try:
            import onnx
            from onnxsim import simplify
            m, ok = simplify(onnx.load(str(out)))
            if ok:
                onnx.save(m, str(out))
                print("[export] onnxsim 化简完成")
        except ImportError:
            print("[export] 未安装 onnxsim，跳过化简（可选：pip install onnxsim）")
        except Exception as e:
            print("[export] onnxsim 失败，保留原始导出: %s" % e)

    if cfg["verify"]:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = torch.randn(1, C, H, W)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        print("[verify] 输出形状 %s | 最大绝对误差 %.2e | argmax 一致率 %.4f" % (
            got.shape, np.abs(ref - got).max(), (ref.argmax(1) == got.argmax(1)).mean()))

    info = dict(meta, onnx=out.name, input_shape=[1, C, H, W], output="logits (未做 softmax/argmax，后处理在主机端)")
    with open(out.parent / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[export] 完成。meta: %s" % (out.parent / "model_meta.json"))


if __name__ == "__main__":
    main()