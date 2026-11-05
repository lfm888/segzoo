# SOTA 模型（`sota/`）

`sota/` 放的是不属于 smp 的网络（LiteSeg、TransUNet）。通过 `sota_model` 字段启用，**不需要改训练代码**，
训练、预测、导出、sweep 的流程和 smp 模型完全一样。

## 已支持的模型

| `sota_model` | 结构 | `sota_backbone` | 输入要求 |
|---|---|---|---|
| `liteseg` | DASPP + 轻量解码器 | **必填**：`resnet18/34/50/101/152`、`mobilenet_v2` | 32 的倍数，任意长宽比 |
| `transunet` | R50 + ViT-B/16，**结构固定** | **必须留空** | **正方形**，16 的倍数（如 224） |

`sota_act`（激活函数）只对 LiteSeg 有效，TransUNet 忽略。

## 使用

在 `train.py` 的 DEFAULTS 里设置（或命令行覆盖）：

```python
"sota_model": "liteseg",      # 非空 = 使用 sota/ 下的模型，此时 arch / encoder 不参与建模
"sota_backbone": "resnet18",  # 骨干；结构固定的模型（transunet）留空
"sota_act": "relu",           # 激活函数：relu / leakyrelu / silu / hardswish ...
```

```bash
# LiteSeg：有可选骨干，可以扫骨干
python train.py --dataset <数据集> --sota_model liteseg --sota_backbone resnet18
python train.py --dataset <数据集> --sota_model liteseg --sota_backbone mobilenet_v2 --in_channels 1
python sweep.py --dataset <数据集> --sota_model liteseg --encoders resnet18,resnet34,mobilenet_v2

# TransUNet：固定结构，骨干留空，输入必须正方形
python train.py --dataset <数据集> --sota_model transunet --sota_backbone "" --img_size 224,224
python sweep.py --dataset <数据集> --sota_model transunet          # 只跑 1 组
```

### 规则与报错

- `sota_model` 为空（默认）时完全走原来的 smp encoder-decoder，行为不变。
- **结构固定的模型填了骨干**：打印提示并忽略，不会报错（`transunet 的结构是固定的…sota_backbone='resnet18' 已忽略`）。
- **有可选骨干的模型填错/留空**：直接报错，并列出可选项。
- `transunet` + 非正方形输入：报错说明只支持正方形。
- 名字写错会列出所有可用模型。
- checkpoint 的 `meta` 里记录了 `sota` 字段（含 `backbone: ""`），`predict.py` / `export_onnx.py` 据此重建。
- 运行目录名：`liteseg_resnet18_128x128`、`transunet_224x224`（骨干为空时不会出现多余下划线）。

## TransUNet

源码：Beckschen/TransUNet（Apache-2.0，`sota/transunet/LICENSE`）。`vit_seg_*.py` 保持上游文件名与结构，方便同步上游。

### 预训练权重

R50+ViT-B/16 权重约 380MB，**不随代码分发**，需要自己下载后放到：

```
sota/transunet/checkpoints/R50+ViT-B_16.npz
```

下载地址见上游 README（Google Drive）。放好后 `--encoder_weights imagenet`（默认）会自动加载。
文件不存在时打印提示并从零开始训练。

### 迁移时处理的问题

1. 上游 `get_b16_config()` 缺少 `n_skip` / `skip_channels`，只能用 `get_r50_b16_config()`。
2. `config.patches.grid` 必须跟随 `img_size` 设为 `(img_size//16, img_size//16)`，否则 patch 大小算成 0。
3. 解码器用 `int(np.sqrt(n_patch))` 还原特征图，所以只支持正方形输入、且边长是 16 的倍数。
4. 上游用 `os.path.join` 拼 `.npz` 权重 key，**在 Windows 下必然 KeyError**（源码原样运行会失败）。
   迁入后改为固定用 `/`，Linux 下行为与原来相同。
5. 单通道输入：上游在 forward 里自动复制成 3 通道，无需改网络。

### 验证结果

| 项目 | 结果 |
|---|---|
| 与源码逐数值对比（同一权重） | 输出最大差 **0**（`128x128` 单通道） |
| 参数量 | 105.17M，与上游一致 |
| 预训练权重加载 | 成功（源码在 Windows 下 KeyError） |
| 1 轮 demo 训练（预训练 vs 随机） | mDice **0.863 vs 0.137** |
| ONNX 导出 | argmax 一致率 1.0000 |
| 下游链路 | `predict.py`（torch / onnx 结果一致）、`sweep.py` 均通过 |

### 注意事项

- 模型很大：401MB FP32，导出 ONNX 后体积相近，**RKNN 转换基本不适合**（主要在 GPU 上用）。
- 输入 224 时 ViT 位置编码会从预训练的 24x24 网格插值到 14x14（日志 `grid-size from 24 to 14`，正常）。

## LiteSeg

源码来自 `python_model_train`（作者 zh320），论文 https://arxiv.org/abs/1912.06683 。

与源项目默认设置不同的地方：

| 项目 | 源项目 | segzoo |
|---|---|---|
| 输入归一化 | 原始 0–255，不归一化 | 由 `norm` 决定（默认 `auto`：预训练时用 ImageNet 标准化） |
| 骨干预训练 | 不使用 | 由 `encoder_weights` 决定（默认 `imagenet`，离线时自动退回随机初始化） |
| 学习率 | 统一 | 预训练骨干用 `lr × encoder_lr_mult`（默认 0.1），与 smp 模型一致 |

要和源项目完全一致地复现：

```bash
python train.py --sota_model liteseg --encoder_weights none --norm custom --mean 0 --std 1
```

### 迁移时的改动（已验证）

- **结构与权重完全一致**：同一份权重下与源码输出最大差为 0（resnet18 单通道、mobilenet_v2 三通道均已验证）。
- **单通道 mobilenet_v2 修复**：源码只会替换 `backbone.conv1`，而 MobileNetV2 没有这个属性，
  `in_channels=1` 时直接 `AttributeError`。现在改为替换骨干的第一个卷积，两种骨干都适用。
- 去掉上游的模型注册表，改为统一签名：`liteseg(num_classes, in_channels, backbone, act_type, pretrained, img_size)`。
- 预训练权重改用 torchvision 新版的 `weights=` 参数。

### 注意事项

- **RKNN 转换未验证**：LiteSeg 用到了 `bilinear + align_corners=True` 上采样和全局平均池化，需在 Linux 上实测。
- LiteSeg 只使用骨干的第 2 级（8x）和第 4 级（32x）特征。

## 新增一个 SOTA 模型

1. 建目录 `sota/<模型名>/`，实现构建函数（签名固定）：
   ```python
   def my_model(num_classes, in_channels=1, backbone="", act_type="relu", pretrained=False,
                img_size=None) -> nn.Module:
       ...  # forward 输出未经 softmax 的 logits，形状 (B, num_classes, H, W)
   ```
2. 在 `sota/__init__.py` 里登记两项：
   ```python
   from .my_model import my_model
   SOTA_MODELS = {..., "my_model": my_model}
   SOTA_INFO = {..., "my_model": dict(desc="一句话描述", backbones=["resnet18"],   # 空列表 = 固定结构
                                      uses_act=True, square=False, divisor=None)}
   ```
3. 训练时 `--sota_model my_model` 即可。

约定：
- 有骨干/解码器之分时，建议提供只读的 `encoder` 属性指向预训练部分，分层学习率才会生效
  （参考 `sota/liteseg/model.py` 与 `sota/transunet/model.py`）。没有该属性时所有参数用同一学习率。
- `SOTA_INFO` 的 `square` / `divisor` 会在训练开始前统一校验输入尺寸。

## 来源与许可

- LiteSeg：zh320（2023/10/15），源自 `python_model_train` 项目。**开源前请确认原项目许可证。**
- TransUNet：Beckschen/TransUNet，Apache-2.0（已附 `sota/transunet/LICENSE`）。
