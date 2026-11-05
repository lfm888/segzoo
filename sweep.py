"""批量对比：在同一数据集、同一划分、同一训练预算下，遍历 架构 x 编码器，输出排行榜。

直接运行：python sweep.py
结果：<runs_dir>/<dataset>/sweep_results.csv（按 mDice 排序，并标出 精度-GFLOPs 的帕累托最优）
其余训练超参（lr、增强、loss 等）沿用 train.py 的 DEFAULTS，改那里即可。
"""
import csv
import time
import traceback

import torch

from models import RKNN_FRIENDLY_ARCHS, RKNN_FRIENDLY_ENCODERS
from train import DEFAULTS as TRAIN_DEFAULTS, run
from utils import abspath, parse_args, to_hw

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {
    "archs": list(RKNN_FRIENDLY_ARCHS),        # 例如 ["unet","fpn","linknet","pan"]
    "encoders": list(RKNN_FRIENDLY_ENCODERS),  # 例如 ["mobilenet_v2","resnet18","resnet34"]
    "epochs": 30,                 # 筛选用的小预算；选出前几名后再用 train.py 加大 epochs 精训
    "skip_done": True,            # 已有 results.json 的组合直接复用，中断后可续跑
    # 下面几项默认直接取自 train.py 的 DEFAULTS（单一来源），需要时在命令行覆盖
    "dataset": TRAIN_DEFAULTS["dataset"],
    "data_root": TRAIN_DEFAULTS["data_root"],
    "img_size": list(TRAIN_DEFAULTS["img_size"]),
    "in_channels": TRAIN_DEFAULTS["in_channels"],
    "batch_size": TRAIN_DEFAULTS["batch_size"],
    "runs_dir": TRAIN_DEFAULTS["runs_dir"],
    "device": TRAIN_DEFAULTS["device"],          # auto / cuda / cpu（无 GPU 时用 cpu 做快速对比）
}
# ===========================================================================


def pareto_flags(rows):
    """精度(mdice)越高、GFLOPs 越低越好；返回每行是否不被其它行支配。"""
    flags = []
    for r in rows:
        dom = False
        for o in rows:
            if o is r or r["gflops"] is None or o["gflops"] is None:
                continue
            if o["mdice"] >= r["mdice"] and o["gflops"] <= r["gflops"] and (
                    o["mdice"] > r["mdice"] or o["gflops"] < r["gflops"]):
                dom = True
                break
        flags.append(not dom)
    return flags


def main():
    sw = parse_args(DEFAULTS, "批量对比 架构 x 编码器")
    base = dict(TRAIN_DEFAULTS)
    for k in ("dataset", "data_root", "img_size", "in_channels", "batch_size", "runs_dir", "epochs", "device"):
        base[k] = sw[k]
    hw = to_hw(base["img_size"])
    out_root = abspath(base["runs_dir"]) / base["dataset"]

    rows, failed = [], []
    combos = [(a, e) for a in sw["archs"] for e in sw["encoders"]]
    t0 = time.time()
    for i, (arch, enc) in enumerate(combos, 1):
        print("\n######## [%d/%d] %s + %s ########" % (i, len(combos), arch, enc))
        cfg = dict(base, arch=arch, encoder=enc)
        done = out_root / ("%s_%s_%dx%d" % (arch, enc, hw[0], hw[1])) / "results.json"
        try:
            if sw["skip_done"] and done.exists():
                import json
                res = json.load(open(done, encoding="utf-8"))
                print("[skip] 已有结果，mDice=%s" % res["mdice"])
            else:
                res = run(cfg)
            rows.append(res)
        except Exception as e:  # 某些组合可能不支持该输入尺寸/编码器，记录后继续
            traceback.print_exc()
            failed.append((arch, enc, "%s: %s" % (type(e).__name__, e)))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = [r for r in rows if r.get("mdice") is not None]
    rows.sort(key=lambda r: r["mdice"], reverse=True)
    flags = pareto_flags(rows)

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "sweep_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "arch", "encoder", "mdice", "miou", "params_M", "gflops", "latency_ms", "pareto", "run_dir"])
        for k, (r, fl) in enumerate(zip(rows, flags), 1):
            w.writerow([k, r["arch"], r["encoder"], r["mdice"], r["miou"], r["params_m"], r["gflops"],
                        r["latency_ms"], int(fl), r["run_dir"]])

    print("\n" + "=" * 78)
    print("%-3s %-10s %-14s %-8s %-8s %-9s %-9s %s" % ("#", "arch", "encoder", "mDice", "mIoU", "params(M)", "GFLOPs", "pareto"))
    for k, (r, fl) in enumerate(zip(rows, flags), 1):
        g = "%.2f" % r["gflops"] if r["gflops"] is not None else "-"
        print("%-3d %-10s %-14s %-8.4f %-8.4f %-9.2f %-9s %s" % (
            k, r["arch"], r["encoder"], r["mdice"], r["miou"], r["params_m"], g, "*" if fl else ""))
    for a, e, msg in failed:
        print("[failed] %s + %s -> %s" % (a, e, msg))
    print("总耗时 %.1f 分钟，表格已保存: %s" % ((time.time() - t0) / 60, csv_path))
    print("提示：带 * 的是 精度-计算量 帕累托最优；最终请再看 RKNN 量化后的精度与板上延迟。")


if __name__ == "__main__":
    main()
