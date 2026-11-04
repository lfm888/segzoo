"""模型构建 / 加载 / 统计。所有脚本共用，不含命令行参数。"""
import time

import torch
import segmentation_models_pytorch as smp

# 对 RKNN 比较友好的候选（纯卷积为主）。sweep.py 的 DEFAULTS 会引用，可自行增删。
# smp 支持的其它架构：unetplusplus, manet, deeplabv3, deeplabv3plus, pspnet, upernet, segformer, dpt ...
RKNN_FRIENDLY_ARCHS = ["unet", "fpn", "linknet", "pan"]
RKNN_FRIENDLY_ENCODERS = ["mobilenet_v2", "resnet18", "resnet34"]


def build_model(arch, encoder, in_channels, num_classes, encoder_weights="imagenet"):
    """输出为未经 softmax 的 logits（B, num_classes, H, W）。
    in_channels=1 时 smp 会把预训练第一层卷积权重按通道求和，仍可使用 ImageNet 预训练。"""
    kw = dict(encoder_name=encoder, in_channels=in_channels, classes=num_classes)
    try:
        return smp.create_model(arch, encoder_weights=encoder_weights, **kw)
    except Exception as e:  # 常见原因：没网下载不了预训练权重
        if encoder_weights:
            print("[warn] 加载预训练权重 %r 失败（%s: %s），改为随机初始化" % (encoder_weights, type(e).__name__, e))
            return smp.create_model(arch, encoder_weights=None, **kw)
        raise


def count_params(model):
    """参数量，单位 M"""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gflops(model, shape, device):
    """单次前向 GFLOPs（= 2*MACs）。需要 torch>=2.1，失败返回 None。"""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        x = torch.randn(*shape, device=device)
        with FlopCounterMode(display=False) as fc:
            model(x)
        return fc.get_total_flops() / 1e9
    except Exception:
        return None


@torch.no_grad()
def benchmark_ms(model, shape, device, warmup=10, iters=50):
    """batch=1 的平均前向耗时(ms)。注意这只是本机 GPU/CPU 的相对参考，不等于 NPU 耗时。"""
    model.eval()
    x = torch.randn(*shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def load_checkpoint(path, device="cpu"):
    """读取 train.py 保存的 best.pth / last.pth，返回 (model, meta)。"""
    ck = torch.load(str(path), map_location="cpu")
    meta = ck["meta"]
    model = build_model(meta["arch"], meta["encoder"], meta["in_channels"], meta["num_classes"], None)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), meta
