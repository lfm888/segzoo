# segzoo

基于 [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) 的医学图像分割快速验证框架。

- **换数据集、换模型都只改一个参数**：同一份数据、同一份划分，快速比较 `架构 × 编码器`，也可直接用 LiteSeg / TransUNet 等 SOTA 模型
- **固定划分**：train / val / test 一次生成。训练只看 val，test 留给最终评估
- **自动识别标签**：labelme `.json`、单通道索引 png、多通道 one-hot png
- **可视化**：预测图一张看完原图、分割结果和真值轮廓；训练自动生成曲线图
- **可部署**：归一化在模型外部完成（ONNX 不含归一化）；预处理实现在 `inference/`，与 C++ 逐行对应并附 golden 向量对拍

## 快速开始

### 1. 安装

```bash
conda create -n segzoo python=3.10 -y
conda activate segzoo
# GPU（CUDA 12.1）；CPU 环境把 --index-url 那一段去掉即可
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python check_env.py                     # 自检：依赖、GPU、前向、ONNX 导出
```

Windows 也可以直接运行 `setup_env.ps1` 一键创建环境。

### 2. 冒烟测试（不需要自己的数据）

```bash
python make_demo_data.py                # 生成三个合成数据集 _demo_*
python train.py --dataset _demo_mask --img_size 128,128 --epochs 3
```

### 3. 用自己的数据

把数据放成下面的结构（格式详见 [docs/data.md](docs/data.md)）：

```
data/<数据集名>/
├── train_dataset/     原图
└── labels/            标签（mask 图或 labelme .json）
```

```bash
python inspect_data.py                  # 查看标签形式、类别数、像素占比
python dataset.py --dataset data/<数据集名>   # 生成 split.txt 与 label_info.txt（可选，train.py 会自动生成）
python train.py --dataset <数据集名> --arch unet --encoder resnet18 --img_size 224,224 --epochs 50
```

### 4. 使用 SOTA 模型（LiteSeg / TransUNet）

`sota/` 文件夹里的模型不走 smp，设置 `sota_model` 即可，其余流程不变：

```bash
# LiteSeg：有可选骨干
python train.py --dataset <数据集名> --sota_model liteseg --sota_backbone resnet18

# TransUNet：结构固定，骨干留空；输入必须正方形（16 的倍数）
python train.py --dataset <数据集名> --sota_model transunet --sota_backbone "" --img_size 224,224
```

| `sota_model` | `sota_backbone` | 输入要求 |
|---|---|---|
| `liteseg` | 必填（`resnet18/34/50/101/152`、`mobilenet_v2`） | 32 的倍数 |
| `transunet` | **必须留空** | 正方形、16 的倍数 |

TransUNet 需要自行下载 R50+ViT-B/16 权重（约 380MB）放到 `sota/transunet/checkpoints/`，
没有也能从零训练。

详见 [docs/sota.md](docs/sota.md)。

### 5. 对比与评估

```bash
python sweep.py --dataset <数据集名> --epochs 30     # 批量对比 架构 × 编码器，输出排行榜
python predict.py                                    # 在 test 段做最终评估，输出指标与可视化
```

### 6. 导出部署

```bash
python export_onnx.py                   # best.pth → model.onnx（+ model_meta.json，含预处理规格）
python predict.py --backend onnx        # onnxruntime 再评估一遍，确认前后处理一致
python predict.py --backend dnn         # OpenCV DNN（与 C++ 部署同一套算子）
python inference/preprocess.py          # 预处理自检 + 生成 golden 向量（给 C++ 对拍）
```

三种后端（`torch` / `onnx` / `dnn`）共用同一份预处理，指标应几乎一致。
只要有 ONNX + `model_meta.json`（不需要 checkpoint）就能验证：
`python predict.py --onnx out/model.onnx --backend dnn`

ONNX 的输入是**已经归一化**的 `1xCxHxW float32`，C++ 侧照
[docs/preprocess.md](docs/preprocess.md) 的五个步骤实现即可（每步都有对应的 OpenCV 调用）。

## 目录约定

| 位置 | 内容 |
|---|---|
| `data/<数据集名>/train_dataset/` | 原图 |
| `data/<数据集名>/labels/` | 标签 |
| `data/<数据集名>/split.txt` | 自动生成的 train/val/test 划分，可手工修改 |
| `data/<数据集名>/label_info.txt` | 自动生成的标签扫描缓存 |
| `data/<数据集名>/classes.txt` | 可选，类别名，每行一个 |
| `runs/<数据集名>/<arch>_<encoder>_<H>x<W>/` | 训练输出：权重、日志、曲线、可视化（SOTA 模型为 `<sota_model>_<backbone>_<H>x<W>`，骨干为空时如 `transunet_224x224`） |
| `sota/` | 非 smp 的 SOTA 模型（LiteSeg、TransUNet），每个模型一个子目录 |
| `inference/` | 统一预处理（纯 numpy+cv2，与 C++ 逐行对应），训练与推理共用 |

## 脚本一览

| 脚本 | 作用 |
|---|---|
| `make_demo_data.py` | 生成合成数据，用于冒烟测试 |
| `check_env.py` | 环境自检 |
| `inspect_data.py` | 扫描 `data/` 下所有数据集的标签形式、类别和尺寸 |
| `dataset.py` | 只生成 `split.txt` 和 `label_info.txt`，不训练 |
| `train.py` | 训练单个模型，输出权重、日志、曲线和可视化 |
| `sweep.py` | 批量训练 架构 × 编码器，输出排行榜与帕累托 |
| `predict.py` | 预测 / 最终评估 / **ONNX 验证**（torch、onnxruntime、OpenCV DNN 三种后端） |
| `export_onnx.py` | 导出 ONNX 并校验 |

每个脚本的参数都集中在文件顶部的 `DEFAULTS` 字典里（`dataset.py` 为 `PREP_DEFAULTS`）。可以直接改默认值，也可以在命令行覆盖：

```bash
python train.py --epochs 100 --lr 5e-4 --loss dice
python train.py --acc_exclude_bg false            # 布尔参数传 true / false
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/data.md](docs/data.md) | 数据格式、标签编码、类别名、标签缓存 |
| [docs/split.md](docs/split.md) | train / val / test 划分规则与自动重划 |
| [docs/outputs.md](docs/outputs.md) | 训练输出、日志字段、指标与曲线图说明 |
| [docs/predict.md](docs/predict.md) | 三种推理后端、只验证 ONNX、诊断输出、可视化与最终评估 |
| [docs/sota.md](docs/sota.md) | SOTA 模型（LiteSeg / TransUNet）的用法、迁移说明、如何新增模型 |
| [docs/preprocess.md](docs/preprocess.md) | 统一预处理接口、C++ 对应关系、golden 向量对拍 |

## 常见问题

**验证集 loss 出现几十？**
通常是模型早期塌缩到单一类别，加上 CE 对“自信错分”的惩罚没有上限。可以加 `--label_smoothing 0.05`。日志出现 `[!] 模型可能塌缩` 时也是这个原因。

**Acc 接近 1，但分割效果很差？**
前景占比很小时，全部像素准确率会被背景拉高。主指标默认使用前景像素准确率 `Acc(fg)`，同时请看各类 Dice。

**某个类的 Dice 很低？**
先检查标签：多通道 mask 中重叠的类是否被后面的通道覆盖了。`inspect_data.py` 会显示通道重叠比例。

**改了划分比例，划分没有变？**
划分文件 `split.txt` 会记录生成时的比例，比例变化后会自动重划。如果想固定沿用旧划分，加 `--split_check_ratio false`。

**标签很多，每次启动都很慢？**
第一次扫描后会缓存到 `label_info.txt`，之后直接复用。标签变化时会自动重扫。

**没有 GPU？**
加 `--device cpu`，并把 `--img_size` 调小（如 `64,64`）做快速验证。

## 参与贡献

欢迎提 Issue 和 Pull Request。提交前请先运行冒烟测试：

```bash
python make_demo_data.py && python train.py --dataset _demo_mask --img_size 128,128 --epochs 1 --encoder_weights none --device cpu
```
