"""把训练好的 best.pth 导出为 ONNX（固定输入尺寸、静态 shape，便于转 RKNN），并用 onnxruntime 校验。

直接运行：python export_onnx.py        （自动取 runs 下最新的 best.pth）
指定模型：python export_onnx.py --ckpt runs/AIDK/unet_resnet18_288x512/best.pth
输出：同目录下 model.onnx 与 model_meta.json（记录输入尺寸/归一化/类别映射，供 predict、convert_rknn 使用）

预处理（本版说明）：
  * ONNX 里不含归一化，归一化由 inference/preprocess.py 在模型外完成（C++ 部署同样如此）；
  * 校验时的输入不再是 torch.randn，而是 uint8 图像经 preprocess(...) 得到的 blob，
    即与 predict / 部署完全一致的数值分布（像素 /255 后 (x-mean)/std）；
  * model_meta.json 里的 mean/std 统一写成 0~1 量纲并带 norm_units="0-1"（旧 checkpoint 的 0~255 量纲会被自动换算），
    同时写入 preprocess 子字典（predict 优先读它）以及 rknn_mean_values / rknn_std_values（0~255 量纲，直接给 RKNN config 用）。
"""
import json

import numpy as np
import torch

from inference.preprocess import PreprocessSpec, preprocess
from models import load_checkpoint
from utils import abspath, parse_args, resolve_path

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "ckpt": r"E:\lfm\segzoo\runs\inner_circle_dataset\unet_resnet18_64x64\best.pth",       # latest = runs 下修改时间最新的 best.pth；也可写具体路径
    "runs_dir": "runs",
    "out": "",              # 空 = 与 ckpt 同目录的 model.onnx
    "opset": 12,            # RKNN-Toolkit2 对 opset 12 兼容最稳；失败时可试 13
    "simplify": True,       # 用 onnxsim 化简（未安装会自动跳过）
    "verify": True,         # 与 PyTorch 输出对比（输入为 preprocess() 产出的真实分布 blob）
    "verify_images": 3,     # 校验用的合成 uint8 图像张数（含随机噪声图、渐变图）
}
# ===========================================================================


def _verify_images(spec, n, seed=0):
    """合成 n 张 uint8 图（模型色彩：灰度 HxW 或 RGB HxWx3，尺寸取模型输入尺寸），
    第 0 张随机噪声，第 1 张水平渐变，其余为噪声。覆盖 0~255 的完整范围。"""
    H, W = spec.img_size
    c = spec.in_channels
    shape = (H, W) if c == 1 else (H, W, 3)
    rng = np.random.RandomState(seed)
    imgs = []
    for i in range(max(int(n), 1)):
        if i == 1:
            grad = np.tile(np.linspace(0, 255, W, dtype=np.float32), (H, 1)).astype(np.uint8)
            imgs.append(grad if c == 1 else np.stack([grad] * 3, axis=-1))
        else:
            imgs.append(rng.randint(0, 256, size=shape).astype(np.uint8))
    return imgs


def main():
    cfg = parse_args(DEFAULTS, "导出 ONNX")
    ckpt = resolve_path(cfg["ckpt"], cfg["runs_dir"], "best.pth")
    model, meta = load_checkpoint(ckpt, "cpu")
    # 预处理规格：与训练 / predict / C++ 同一份契约（旧 checkpoint 的 0~255 量纲会在这里自动换算为 0~1）
    spec = PreprocessSpec.from_meta(meta)
    C, (H, W) = spec.in_channels, spec.img_size
    out = abspath(cfg["out"]) if cfg["out"] else ckpt.parent / "model.onnx"
    print("[export] %s -> %s  输入 (1,%d,%d,%d)" % (ckpt, out, C, H, W))
    print("[export] 归一化: %s  像素 /255 -> [0,1] 后 (x-mean)/std  mean=%s std=%s" % (
        spec.norm, [round(v, 4) for v in spec.mean], [round(v, 4) for v in spec.std]))

    # 追踪用的 dummy 输入也走 preprocess，保证数值范围与部署一致（静态 shape，不影响导出结构）
    dummy = torch.from_numpy(preprocess(_verify_images(spec, 1)[0], spec, model_color=True))
    model.eval()
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
        in_name = sess.get_inputs()[0].name
        for k, img in enumerate(_verify_images(spec, cfg["verify_images"])):
            blob = preprocess(img, spec, model_color=True)         # 与部署一致的预处理（含范围检查与 /255）
            with torch.no_grad():
                ref = model(torch.from_numpy(blob)).numpy()
            got = sess.run(None, {in_name: blob})[0]
            print("[verify #%d] blob 范围 [%.3f, %.3f] | 输出形状 %s | 最大绝对误差 %.2e | argmax 一致率 %.4f" % (
                k, float(blob.min()), float(blob.max()), got.shape,
                np.abs(ref - got).max(), (ref.argmax(1) == got.argmax(1)).mean()))

    # model_meta.json：把归一化统一成 0~1 量纲写回，predict / convert_rknn 读它
    info = dict(meta, onnx=out.name, input_shape=[1, C, H, W],
                output="logits (未做 softmax/argmax，后处理在主机端)")
    info.update({
        "norm": spec.norm, "norm_units": "0-1",
        "mean": list(spec.mean), "std": list(spec.std),          # 0~1 量纲（像素先 /255）
        "preprocess": spec.to_dict(),                             # PreprocessSpec.from_meta 优先读这一份
        "rknn_mean_values": spec.mean_255(),                      # 0~255 量纲，直接给 RKNN config(mean_values=...) 用
        "rknn_std_values": spec.std_255(),                        # 0~255 量纲，直接给 RKNN config(std_values=...) 用
    })
    with open(out.parent / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[export] 完成。meta: %s" % (out.parent / "model_meta.json"))


if __name__ == "__main__":
    main()
