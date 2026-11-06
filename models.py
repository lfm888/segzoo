"""模型构建 / 加载 / 统计。所有脚本共用，不含命令行参数。"""
import time

import torch
import segmentation_models_pytorch as smp

from sota import build_sota, sota_info

# 对 RKNN 比较友好的候选（纯卷积为主）。sweep.py 的 DEFAULTS 会引用，可自行增删。
# smp 支持的其它架构：unetplusplus, manet, deeplabv3, deeplabv3plus, pspnet, upernet, segformer, dpt ...
RKNN_FRIENDLY_ARCHS = ["unet", "fpn", "linknet", "unetplusplus","deeplabv3plus"]
RKNN_FRIENDLY_ENCODERS = ["timm-mobilenetv3_small_minimal_100", "resnet18", "resnet34", "efficientnet-b0", "resnext50_32x4d", "mit_b0"] #mit仅支持三通道


def sota_spec(cfg, hw=None):
    """cfg["sota_model"] 非空 -> {"name", "backbone", "act"}，使用 sota/ 下的模型；为空 -> None（走 smp）。

    骨干的规则（由 sota/__init__.py 的 SOTA_INFO 决定）：
      · 有可选骨干的模型（如 liteseg）：sota_backbone 必须是其中之一；
      · 结构固定的模型（如 transunet）：没有骨干可选，sota_backbone 必须留空。填了会打印提示并忽略。
    hw 给出时还会检查输入尺寸（如 TransUNet 只支持正方形、16 的倍数）。"""
    name = str(cfg.get("sota_model") or "").strip().lower()
    if not name:
        return None
    info = sota_info(name)                         # 名字不存在会在这里报错，并列出可选模型

    backbone = str(cfg.get("sota_backbone") or "").strip().lower()
    if not info["backbones"]:
        if backbone:
            print("[sota] %s 的结构是固定的（%s），没有可选骨干；sota_backbone=%r 已忽略。"
                  "请把 sota_backbone 留空。" % (name, info["desc"], backbone))
        backbone = ""
    elif backbone not in info["backbones"]:
        raise ValueError("sota_backbone=%r 不适用于 %s，可选: %s"
                         % (cfg.get("sota_backbone"), name, ", ".join(info["backbones"])))

    act = str(cfg.get("sota_act") or "relu").strip().lower() if info["uses_act"] else ""

    if hw is not None:
        h, w = int(hw[0]), int(hw[1])
        if info["square"] and h != w:
            raise ValueError("%s 只支持正方形输入（解码器需要），当前 img_size=%dx%d" % (name, h, w))
        d = info["divisor"]
        if d and (h % d or w % d):
            raise ValueError("%s 要求 img_size 是 %d 的倍数，当前 %dx%d" % (name, d, h, w))

    return {"name": name, "backbone": backbone, "act": act}


def model_desc(arch, encoder):
    """显示用："liteseg + resnet18"；骨干为空时只显示模型名（"transunet"）。"""
    return " + ".join(p for p in (arch, encoder) if p)


def run_name(arch, encoder, h, w):
    """运行目录名："liteseg_resnet18_128x128"；骨干为空时不留多余的下划线（"transunet_224x224"）。"""
    return "_".join([p for p in (arch, encoder) if p] + ["%dx%d" % (int(h), int(w))])


def build_model(arch, encoder, in_channels, num_classes, encoder_weights="imagenet", sota=None, img_size=None):
    """输出为未经 softmax 的 logits（B, num_classes, H, W）。
    sota 为 None 时用 smp 的 arch + encoder 搭建；为 sota_spec() 的返回值时用 sota/ 下的模型
    （img_size 为正方形边长，TransUNet 需要）。
    in_channels=1 时 smp 会把预训练第一层卷积权重按通道求和，仍可使用 ImageNet 预训练。"""
    if sota:
        pretrained = bool(encoder_weights) and str(encoder_weights).lower() != "none"
        return build_sota(sota["name"], num_classes, in_channels=in_channels, backbone=sota["backbone"],
                          act_type=sota.get("act", "relu"), pretrained=pretrained, img_size=img_size)

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
    """读取 train.py 保存的 best.pth / last.pth，返回 (model, meta)。
    meta 里有 sota 字段时按 sota 模型重建，否则按 smp 的 arch / encoder 重建（旧 checkpoint 也兼容）。"""
    ck = torch.load(str(path), map_location="cpu")
    meta = ck["meta"]
    model = build_model(meta["arch"], meta["encoder"], meta["in_channels"], meta["num_classes"], None,
                        sota=meta.get("sota"), img_size=int(meta["img_size"][0]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), meta
