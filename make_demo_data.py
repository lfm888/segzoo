"""生成两个合成小数据集，用来冒烟测试整条流程（不依赖你的真实数据）。
  data/_demo_mask      ：3 类 mask 图像（像素值 0/128/255，测试自动映射）
  data/_demo_labelme   ：labelme json 单类别多边形
"""
import json
from pathlib import Path

import cv2
import numpy as np

from utils import abspath, imwrite_unicode, parse_args

# ============================ 所有参数集中在这里 ============================
DEFAULTS = {"data_root": "data", "n": 24, "size": 128, "seed": 0}
# ===========================================================================


def main():
    cfg = parse_args(DEFAULTS, "生成合成测试数据")
    rng = np.random.RandomState(cfg["seed"])
    S = cfg["size"]
    for name in ("_demo_mask", "_demo_labelme"):
        for sub in ("train_dataset", "labels"):
            (abspath(cfg["data_root"]) / name / sub).mkdir(parents=True, exist_ok=True)
    for i in range(cfg["n"]):
        cx, cy = rng.randint(S // 3, 2 * S // 3, 2)
        a, b = rng.randint(S // 6, S // 3, 2)
        img = (rng.rand(S, S) * 40 + 30).astype(np.float32)
        m = np.zeros((S, S), np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (int(a), int(b)), 0, 0, 360, 1, -1)
        cv2.ellipse(m, (int(cx), int(cy)), (max(2, int(a) // 2), max(2, int(b) // 2)), 0, 0, 360, 2, -1)
        img[m == 1] += 80
        img[m == 2] += 150
        img = np.clip(img, 0, 255).astype(np.uint8)
        root = abspath(cfg["data_root"])
        # 3 类 mask：像素值用 0/128/255
        imwrite_unicode(root / "_demo_mask" / "train_dataset" / ("%03d.png" % i), img)
        imwrite_unicode(root / "_demo_mask" / "labels" / ("%03d.png" % i), np.array([0, 128, 255], np.uint8)[m])
        # labelme：椭圆外轮廓近似成多边形
        t = np.linspace(0, 2 * np.pi, 40)
        pts = np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], 1).tolist()
        imwrite_unicode(root / "_demo_labelme" / "train_dataset" / ("%03d.png" % i), img)
        js = {"version": "5.0.0", "imagePath": "%03d.png" % i, "imageHeight": S, "imageWidth": S,
              "shapes": [{"label": "obj", "points": pts, "shape_type": "polygon", "flags": {}}]}
        (root / "_demo_labelme" / "labels" / ("%03d.json" % i)).write_text(json.dumps(js), encoding="utf-8")
    print("已生成合成数据: %s/_demo_mask, %s/_demo_labelme" % (cfg["data_root"], cfg["data_root"]))


if __name__ == "__main__":
    main()
