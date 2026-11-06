# SegZoo

一个轻量、高效且易于扩展的 PyTorch 图像分割训练与推理框架。支持经典 SMP（Segmentation Models PyTorch）模型，并内置了独立的 SOTA 前沿模型扩展（如 LiteSeg、TransUNet、YOLO11 语义分割）。

- **换数据集、换模型都只改一个参数**：同一份数据、同一份划分，快速比较 `架构 × 编码器`，也可直接用 LiteSeg / TransUNet 等 SOTA 模型
- **固定划分**：train / val / test 一次生成。训练只看 val，test 留给最终评估
- **自动识别标签**：labelme `.json`、单通道索引 png、多通道 one-hot png
- **可视化**：预测图一张看完原图、分割结果和真值轮廓；训练自动生成曲线图
- **可部署**：归一化在模型外部完成（ONNX 不含归一化）；预处理实现在 `inference/`，与 C++ 逐行对应并附 golden 向量对拍
## 🌟 特点

* **双模支持**：既支持标准 SMP 架构（UNet, FPN, DeepLabV3+ 等），也支持独立实现的 SOTA 模型（如 `sota/liteseg`、`sota/transunet`、`sota/yolo11`）。
* **灵活的权重加载**：支持从零训练，或指定加载官方预训练权重、微调 `.pt` / `.pth` 权重。
* **开箱即用**：集成了单模型训练、日志记录、自动混淆矩阵评估、指标曲线绘制、模型复杂度统计以及验证集可视化。

## 📂 项目结构

```
segzoo/
│
├── dataset.py            # 数据集加载与处理（支持 Labelme JSON、单通道索引、多通道 Mask）
├── models.py             # 模型构建、参数量统计、FLOPs 计算及检查点读取
├── train.py              # 单模型训练入口
├── utils.py              # 常用工具函数（指标计算、可视化渲染、曲线绘制等）
│
├── sota/                 # SOTA 前沿模型扩展
│   ├── __init__.py       # SOTA 模型统一注册与入口
│   ├── liteseg/          # LiteSeg 模型实现
│   ├── transunet/        # TransUNet 模型实现
│   └── yolo11/           # YOLO11 语义分割模型实现
│       ├── model.py      # YOLO11 骨干+颈部+分割头独立实现
│       └── checkpoints/  # 放置 yolo11*-seg.pt 预训练权重
│
└── runs/                 # 训练输出目录（自动保存 best.pth, log.csv, curves.png 等）
```

## 🚀 快速开始

### 1. 数据准备

将你的数据集放在 `data/<dataset_name>/` 目录下：

* `<dataset_name>/train_dataset/`：输入图像
* `<dataset_name>/labels/`：分割标签
* 类别文件：可自动扫描，或在目录下提供 `classes.txt`。

### 2. 训练模型

#### 方法 A：使用标准 SMP 模型（如 UNet + ResNet18）

```bash
python train.py --arch unet --encoder resnet18 --epochs 50 --batch_size 8
```

#### 方法 B：使用内置的 SOTA 模型（如 YOLO11 语义分割）

框架支持通过 `sota_model` 指定前沿模型，并通过 `encoder_weights`（或对应权重参数）加载本地 `.pt` 权重：

1. **准备权重**：将 YOLO11 的分割权重（如 `yolo11n-seg.pt`）放入 `sota/yolo11/checkpoints/` 目录下。
2. **运行训练**：

   ```bash
   python train.py --sota_model yolo11 --sota_backbone n --sota_act silu --encoder_weights imagenet --epochs 50
   ```

   *说明：`encoder_weights="imagenet"` 或 `True` 时，代码会自动到 `sota/yolo11/checkpoints/` 寻找对应档位的 `.pt` 文件并自动加载骨干与颈部权重。*

## ⚙️ 核心参数说明

可以在 `train.py` 的 `DEFAULTS` 字典中修改默认配置，或通过命令行直接覆盖：

| 参数名 | 默认值 | 说明 | 
| ----- | ----- | ----- | 
| `--data_root` | `"data"` | 数据集根目录 | 
| `--dataset` | `"inner_circle_dataset"` | 数据集文件夹名称 | 
| `--img_size` | `[64, 64]` | 输入图像尺寸 `[H, W]`（需为 32 的倍数） | 
| `--in_channels` | `1` | 输入通道数（1=灰度图，3=RGB） | 
| `--sota_model` | `"liteseg"` | 启用 SOTA 模型（如 `yolo11`、`transunet`、`liteseg`），留空则使用 SMP 模型 | 
| `--sota_backbone` | `"resnet18"` | SOTA 模型骨干或档位（如 YOLO11 的 `n`/`s`/`m`/`l`/`x`） | 
| `--encoder_weights` | `"imagenet"` | 预训练权重（`imagenet`、`none` 或指定 `.pt`/`.pth` 文件路径） | 
| `--epochs` | `20` | 训练总轮数 | 
| `--batch_size` | `8` | 批次大小 | 
| `--lr` | `1e-3` | 学习率 | 

## 📊 训练产物

每次训练会在 `runs/<dataset>/<model_name>/` 下生成以下产物：

* `best.pth`：验证集 `mDice` 最高时的模型权重与元数据
* `last.pth`：最后一轮的训练权重
* `log.csv`：每轮的训练/验证 Loss 与各项分割指标（Dice、IoU、准确率等）
* `curves.png`：自动绘制的 Loss 与指标变化趋势图
* `vis/`：验证集可视化对比图（虚线 = GT 真值，色块 = 模型预测）

## ❓ 常见问题（FAQ）

1. **验证集 loss 出现几十？**
   * 通常是模型早期塌缩到单一类别，加上 CE 对“自信错分”的惩罚没有上限。可以加 `--label_smoothing 0.05`。日志出现 `[!] 模型可能塌缩` 时也是这个原因。

2. **Acc 接近 1，但分割效果很差？**
   * 前景占比很小时，全部像素准确率会被背景拉高。主指标默认使用前景像素准确率 `Acc(fg)`，同时请看各类 Dice。

3. **某个类的 Dice 很低？**
   * 先检查标签：多通道 mask 中重叠的类是否被后面的通道覆盖了。`inspect_data.py` 会显示通道重叠比例。

4. **改了划分比例，划分没有变？**
   * 划分文件 `split.txt` 会记录生成时的比例，比例变化后会自动重划。如果想固定沿用旧划分，加 `--split_check_ratio false`。

5. **标签很多，每次启动都很慢？**
   * 第一次扫描后会缓存到 `label_info.txt`，之后直接复用。标签变化时会自动重扫。

6. **没有 GPU？**
   * 加 `--device cpu`，并把 `--img_size` 调小（如 `64,64`）做快速验证。

7. **输入图片通道数为 1（灰度图）时，如何加载 ImageNet / YOLO11 预训练权重？**
   * 框架内部会在第一层卷积（`conv1`）自动将预训练的 3 通道权重按通道求和或取均值，因此即使 `in_channels=1` 也可以无缝加载 ImageNet 预训练权重。

8. **加载 YOLO11 的 `.pt` 检查点提示报错或读取失败？**
   * 项目使用了原生的纯 PyTorch 解包机制。如果你的环境中未安装 `ultralytics`，框架会自动尝试使用兼容模式解析 `.pt` 里的权重字典。若仍有问题，建议使用官方包转存为纯 `state_dict` 文件：

     ```python
     from ultralytics import YOLO
     import torch
     torch.save(YOLO("yolo11n-seg.pt").model.state_dict(), "yolo11n-seg.state_dict.pth")
     ```

9. **输入尺寸（`img_size`）有什么限制？**
   * 一般基于卷积的网络（UNet、FPN、YOLO11）要求高宽为 32 的倍数；部分 Transformer 架构（如 TransUNet）要求输入为正方形且为 16 或 32 的倍数。

## 🤝 参与贡献

欢迎对本框架提出改进建议或贡献代码！你可以通过以下步骤参与：

1. **Fork** 本仓库并 Clone 到本地。
2. 创建你的特性分支 (`git checkout -b feature/AmazingSOTA`)。
3. 如果你想添加新的 SOTA 模型：
   * 在 `sota/` 目录下新增你的模型实现文件夹。
   * 在 `sota/__init__.py` 中登记你的模型与 `SOTA_INFO` 信息。
4. 提交更改 (`git commit -m 'Add some AmazingSOTA'`)。
5. 推送到分支 (`git push origin feature/AmazingSOTA`)。
6. 提交 **Pull Request**。

欢迎提交 Issue 反馈 Bug 或讨论新功能需求！