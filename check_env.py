"""环境自检：依赖版本、GPU、smp 构建模型、前向、导出 ONNX 并用 onnxruntime 跑一遍。"""
import sys
import tempfile
from pathlib import Path


def main():
    import cv2
    import numpy as np
    import torch
    import timm
    import segmentation_models_pytorch as smp
    import onnx
    import onnxruntime as ort

    print("python        :", sys.version.split()[0])
    print("torch         :", torch.__version__, "| cuda 可用:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    print("smp / timm    :", smp.__version__, "/", timm.__version__)
    print("numpy / cv2   :", np.__version__, "/", cv2.__version__)
    print("onnx / ort    :", onnx.__version__, "/", ort.__version__)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.create_model("unet", encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=3).to(dev).eval()
    x = torch.randn(1, 1, 64, 64, device=dev)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 3, 64, 64), y.shape
    print("前向测试      : OK", tuple(y.shape))

    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "t.onnx")
        torch.onnx.export(model.cpu(), torch.randn(1, 1, 64, 64), p, opset_version=12,
                          input_names=["input"], output_names=["logits"])
        out = ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(None, {"input": np.random.randn(1, 1, 64, 64).astype(np.float32)})[0]
        assert out.shape == (1, 3, 64, 64)
    print("ONNX 导出测试 : OK")
    print("ALL OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
