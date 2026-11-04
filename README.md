# segzoo：基于 segmentation_models.pytorch 的分割模型库

目标：对任意分割数据集，系统地对比 架构 × 编码器，在精度与可部署性（ONNX → RKNN）之间选最合适的设计。

## 目录约定
```
data/<数据集名>/train_dataset/   图像
data/<数据集名>/labels/          标签：labelme 的 .json，或 mask 图像（按文件名主干匹配）
```
`data_root` 默认是本目录下的 `data`，也可在 `train.py` 的 DEFAULTS 里写绝对路径。

## 所有脚本：参数集中在文件顶部 DEFAULTS，直接运行；命令行 --xxx 可覆盖
| 脚本 | 作用 |
|---|---|
| `check_env.py` | 环境自检（依赖、GPU、前向、ONNX 导出） |
| `make_demo_data.py` | 生成合成数据，冒烟测试流程 |
| `train.py` | 训练单个模型，保存 best.pth / 日志 / 可视化 / results.json |
| `sweep.py` | 批量对比 架构×编码器，输出排行榜与帕累托 |
| `export_onnx.py` | best.pth → 静态 shape 的 ONNX（opset 12）并校验 |
| `predict.py` | 无需标签的批量预测可视化（torch 或 onnx 后端） |
| `convert_rknn.py` | ONNX → RKNN（仅 Linux/WSL2，需 rknn-toolkit2） |

## 典型流程
```
python make_demo_data.py
python train.py --dataset _demo_mask --img_size 128,128 --epochs 3        # 冒烟
python train.py                       # 在 DEFAULTS 指定的数据集上训练
python sweep.py                       # 批量对比
python export_onnx.py && python predict.py --backend onnx
# Linux 上：python convert_rknn.py
```
