# 预测、最终评估与 ONNX 验证（`predict.py`）

`predict.py` 同时承担三件事：批量预测、在 test 段做最终评估、验证导出的 ONNX 是否可用。
原 `predict_onnx.py` 已合并进来，不再单独维护。

## 三种推理后端

三者**共用同一份预处理**（[`inference/preprocess.py`](preprocess.md)），只是执行引擎不同：

| `--backend` | 引擎 | 用途 |
|---|---|---|
| `torch`（默认） | PyTorch `best.pth` | 与训练完全一致，作为基准 |
| `onnx` / `onnxruntime` / `ort` | onnxruntime | 验证导出的 ONNX 在通用推理引擎下正确 |
| `dnn` / `opencv` / `cv2` | OpenCV DNN (`cv::dnn::readNetFromONNX`) | 与 C++ 部署走同一套算子，**最接近板上行为** |

```bash
python predict.py                                   # PyTorch
python predict.py --backend onnx                    # onnxruntime
python predict.py --backend dnn                     # OpenCV DNN
```

三种后端对同一份数据应给出几乎相同的指标（实测差异仅在 1e-4 量级，来自浮点误差）。
建议的验收顺序：`torch` → `onnx` → `dnn`，指标一致即说明**前后处理与导出都没问题**。

## 只验证 ONNX（不需要 checkpoint）

`model_meta.json` 里已包含预处理规格（in_channels / img_size / mean / std），
所以只要有 ONNX + meta 两个文件就能跑，非常适合验证交付给别人的模型：

```bash
python predict.py --onnx out/model.onnx --backend dnn
python predict.py --onnx D:/deliver/model.onnx --input_dir D:/new_images --backend onnx
```

- 没找到 `model_meta.json` 时会退回从 checkpoint 读规格，并打印警告。
- `--onnx` 留空时，默认用与 ckpt 同目录的 `model.onnx`。

## 输入目录

`--input_dir` 支持三种情况，代码自动识别：

| 传入的目录 | 行为 |
|---|---|
| 不传 | 使用 checkpoint 对应的训练数据集，默认只跑 `split.txt` 中的 **test** 段 |
| 同格式数据集目录（有 `train_dataset/`，可选 `labels/`、`split.txt`） | 有 `split.txt` 时按 `--split` 取子集（默认 test）；有标签时计算指标并画真值虚线 |
| 纯图像目录（只有图片） | 对目录中**所有图像**推理；没有标签时不画虚线、不算指标 |

也可以直接指向 `<数据集>/train_dataset/`，会自动在上一级找 `labels/` 和 `split.txt`。

## 常用命令

```bash
python predict.py                                    # 最新 best.pth，在训练数据集的 test 段评估
python predict.py --input_dir data/new_images        # 纯图像目录：全部推理，只出图
python predict.py --split val                        # 看 val 段
python predict.py --save_vis false                   # 只要指标，不存图（更快）
python predict.py --max_images 8                     # 只跑前 8 张
python predict.py --verbose                          # 打印 blob / logits 的形状与数值范围
python predict.py --timing false                     # 不统计各阶段耗时
python predict.py --complexity false                 # 不打印模型复杂度
python predict.py --progress_bar false               # 不显示进度条
```

> 布尔参数支持 `--flag`（等价 true）和 `--flag false` 两种写法。

## 输出顺序

一次推理依次打印：模型信息 → 预热 → **模型复杂度** → **进度条** → **[timing]** → 指标表。

### 进度条

逐张推理时只显示一行进度（不再每张打一行）：

```
  predict [############------------] 5/10  50%  005.png  0m01s<0m01s
```

- 含文件名、百分比、已用时间与 ETA；非终端环境（重定向到文件）自动改为每 20% 一行。
- `--progress_bar false` 关闭。

### 模型复杂度

在 `[timing]` 之前打印当前推理平台上的模型情况（`--complexity false` 关闭）：

```
============================================================
模型复杂度统计
============================================================
参数总量      : 105,276,211 (105.276 M)
可训练参数    : 105,276,211 (105.276 M)
参数占用空间  : 401.60 MB
参数数据类型分布:
  - torch.float32       105,276,211 个 (105.276 M)  401.60 MB
Buffer 数据类型分布（BN 统计量等，不计入参数量）:
  - torch.float32             2,880 个  0.0110 MB
  - torch.int64                   9 个  0.0001 MB
统计输入尺寸  : (1, 1, 224, 224)  dtype=torch.float32
计算量 MACs   : 29,182,377,984 (29.182 GMACs)
计算量 FLOPs  : 58,364,755,968 (58.365 GFLOPs，FLOPs = 2 x MACs)
统计方式      : torch.utils.flop_counter
FLOPs 占比前几的算子:
  - aten.addmm                   33.294 GFLOPs (57.0%)
  - aten.convolution             23.654 GFLOPs (40.5%)
  - aten.bmm                     1.416 GFLOPs (2.4%)
============================================================
```

- 计算量用**与 ONNX 导出相同的输入尺寸**统计，`--verbose` 时额外打印算子占比。
- `onnx` / `dnn` 后端也打印这份报告（结构与参数量相同，只是执行引擎不同）；用 `--onnx` 单独验证且没有
  checkpoint 时会跳过并提示。
- 实现是 `utils.print_model_complexity`，`train.py --complexity true` 也调用它。

### 详细诊断

`--verbose`（可选 `--verbose true`）只对第一张图打印，用来确认模型真的吃对了输入：

```
[verbose] blob    shape=(1, 1, 128, 128) dtype=float32 range=[-0.765, 0.718]
[verbose] logits  shape=(1, 3, 128, 128) dtype=float32 range=[-0.822, 0.811]
```

- blob 的 range 明显偏离预期 → 预处理（mean/std）不对
- logits 空间尺寸与输入不一致 → 模型或导出有问题
- 输出通道数与 `num_classes` 不符 → 导出时的 classes 不对

### 分阶段耗时

`--timing`（默认开启）统计每张图各阶段平均耗时：

```
[timing] 平均每张：预处理 0.37 ms + 推理 2.65 ms + 后处理 0.56 ms = 3.58 ms（backend=torch, 输入 128x128）
```

推理前会先跑一次预热，避免首次初始化耗时污染统计。

## 输出

- 每张图一张 `<stem>_result.png`：原图 + 半透明分割色块 + 同色虚线边界 + 左上角图例。
- 有标签时，额外输出 `metrics_<split>.json` 并打印指标表（含 `backend`、`model`、`latency_ms_per_image`）。
- 默认输出目录是 `<ckpt 所在目录>/predict`；指定 `--input_dir` 时为 `predict_<目录名>`。

## 可视化约定

| 元素 | 含义 |
|---|---|
| 半透明色块 | 模型预测 |
| 同色虚线 | 轮廓来源，由 `--edge` 决定 |
| 原图背景 | 原分辨率；预测结果最近邻放大后叠加 |

`--edge`：

- `auto`（默认）：有标签就画真值，没有就不画。
- `gt`：画真值轮廓。
- `pred`：画预测自身的轮廓（适合没有标签的数据）。
- `none`：不画虚线。

看图方法：色块与虚线重合 = 预测正确；虚线露在色块外 = 漏检；只有色块没有虚线 = 误检。

其他样式参数：`--alpha`（色块透明度）、`--thickness`（虚线粗细，默认按图像尺寸）、
`--dash` / `--gap`（虚线节奏）、`--legend false`（不画图例）。

## 指标

指标在**模型输入尺寸**上计算：预测结果与最近邻缩小后的真值逐像素比较。
训练、验证、测试使用同一套函数 `utils.seg_metrics`，口径一致。
