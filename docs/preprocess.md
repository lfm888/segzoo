# 统一预处理（`inference/preprocess.py`）

训练、PyTorch 推理、ONNX 推理和 C++ 部署共用**一份**预处理实现。归一化在模型外部完成，
**ONNX 里不含归一化**，所以导出后的 ONNX 输入就是已经归一化好的 `1xCxHxW float32`。

## 为什么放在 `inference/`

`inference/preprocess.py` 只依赖 numpy + cv2，不含 torch，可以逐行翻译成 C++。它是唯一实现，
仓库里其它地方（`dataset.py`、`predict.py`、`train.py`、`utils.py`）都调用它，不再各自拼一遍。

## 流水线（每一步都有对应的 OpenCV 调用）

| 步骤 | 函数 | 做什么 | C++ 等价 |
|---|---|---|---|
| 1 | `read_image` | 读文件 → uint8（3 通道为 **BGR**） | `cv::imread(path, IMREAD_COLOR / IMREAD_GRAYSCALE)` |
| 2 | `to_model_color` | BGR→RGB，或 BGR→灰度 | `cv::cvtColor(..., COLOR_BGR2RGB / COLOR_BGR2GRAY)` |
| 3 | `resize` | 缩放到模型输入尺寸 | `cv::resize(img, img, cv::Size(W,H), 0, 0, interp)` |
| — | *数据增强* | **仅训练时**，插在步骤 3 与 4 之间（输出 float32、0~255 量纲） | 无（推理不做增强） |
| 4a | `to_unit_range` | 像素范围检查 → `/255`，统一到 `[0,1]` 的 float32 | `img.convertTo(f, CV_32F, 1.0/255.0)` |
| 4b | `normalize` | `(x-mean)/std` 逐通道（mean/std 为 0~1 量纲） | `cv::subtract / cv::divide` + `Scalar` |
| 5 | `to_blob` | HWC → `1xCxHxW` 连续内存 | `cv::dnn::blobFromImage(f, 1.0, Size(), Scalar(), false, false, CV_32F)` |

组合入口：

| 函数 | 包含步骤 | 用途 |
|---|---|---|
| `load_model_image(path, c)` | 1+2 | 数据集读图（结果 = 模型色彩、原分辨率） |
| `prepare_image(src, spec, model_color=False)` | 1+2+3 | 得到模型输入尺寸的 uint8 图 |
| `preprocess(src, spec, model_color=False)` | 1+2+3+4+5 | **唯一对外入口**：pth / onnxruntime / OpenCV DNN 直接喂 |

`src` 既可以是文件路径，也可以是 OpenCV 的 BGR / 灰度 uint8 数组。
`model_color=True` 表示传入的数组**已经是模型色彩**（RGB / 灰度，例如 `SegDataset.get_full()` 返回的图），
此时跳过步骤 2，避免二次 `BGR→RGB`。

### 两个必须照抄的细节

1. **resize 的插值规则**：只要宽或高有一边需要**缩小**，两个方向都用 `INTER_AREA`；否则用 `INTER_LINEAR`；
   尺寸本来就一样时不做缩放。C++ 里同样要按这个规则选 `interp`。
2. **归一化不能用 `blobFromImage` 的 mean/scalefactor**：`scalefactor` 只能是标量，而 3 通道的 std 逐通道不同。
   C++ 里要用 `cv::subtract` / `cv::divide` + `cv::Scalar`（顺序是 RGB，因为第 2 步已转 RGB），
   `blobFromImage` 只负责 HWC→NCHW（`swapRB=false`、mean 为空、`scalefactor=1`）。

## 量纲约定（最容易出错的地方）

| 量 | 量纲 | 说明 |
|---|---|---|
| 像素 | uint8 `0~255`，或 float `0~255`（增强后） | `to_unit_range` 检查范围后统一 `/255`；float 越界/含 NaN 直接报错 |
| `mean` / `std` | **0~1**（ImageNet 即 `0.485/0.229`） | `PreprocessSpec` 里写死这个量纲，传 0~255 的值会直接报 `ValueError` |
| RKNN config | 0~255 | 用 `spec.mean_255()` / `spec.std_255()` 换算，**不要手写 ×255** |

归一化只有 `normalize(img, spec)` 一个入口，训练 / 验证 / 推理 / 导出 / 自检全部走它。

归一化方案（`config["norm"]`，也可 `--norm` 覆盖）只有两种：

| `norm` | mean / std | 等价于 |
|---|---|---|
| `imagenet`（默认） | 3 通道 `(0.485,0.456,0.406)` / `(0.229,0.224,0.225)`，灰度取三通道均值近似 | `(x/255 - μ)/σ` |
| `01` | `0` / `1` | 只做 `x/255`，输入范围 `[0,1]` |

## 规格（`PreprocessSpec`）

一份“契约”，训练时写进 checkpoint 的 `meta` 与 `model_meta.json`，推理端只读它：

```python
from inference.preprocess import PreprocessSpec, norm_stats

spec = PreprocessSpec.from_config(cfg)          # 训练端：按 cfg 的 norm / in_channels / img_size 构造
spec = PreprocessSpec.from_meta(meta)           # 推理端：从 best.pth 的 meta / model_meta.json 构造
mean, std = norm_stats("imagenet", 3)           # 手动构造时取预设：0.485... / 0.229...
spec = PreprocessSpec(in_channels=3, img_size=(224, 224), mean=mean, std=std, norm="imagenet")

spec.mean_255(), spec.std_255()                 # 需要 0~255 量纲的下游（RKNN config）用这两个
```

- `mean` / `std` 是 **0~1 量纲**，通道顺序 **RGB**，长度可以是 1（自动广播到所有通道）或 `in_channels`。
- `norm` 记录方案名（`imagenet` / `01` / `custom`），只做标识用；实际数值以 `mean`/`std` 为准。
- 量纲守卫：`mean` 不在 `[0,1]` 或 `std > 1` 时直接报错（否则 `/255` 后再减 `123.675`，整张图会变成常数且不报错）。
- 长度既不是 1 也不是 `in_channels` 时直接报错，不再静默忽略。
- 旧 checkpoint 兼容：`from_meta` 发现没有 `norm_units` 且数值 > 1 时，自动 `/255` 并给出警告。
  新训练 / 新导出的 meta 带有 `"norm_units": "0-1"` 标记。

## 后处理

| 函数 | 做什么 | C++ 等价 |
|---|---|---|
| `argmax_mask(logits)` | `1xCxhxw` → `hxw` uint8 类别图；并列取**下标最小**的类 | 手写循环，用严格大于 `>` 更新最大值 |
| `resize_mask_nearest(mask, orig_hw)` | 类别图最近邻放大回原图尺寸（不产生新类别） | `cv::resize(..., INTER_NEAREST)` |

## 自检 + golden 向量

直接运行本文件会做自检，并生成给 C++ 对拍用的 golden 向量：

```bash
python inference/preprocess.py                                        # 用最新 model_meta.json
python inference/preprocess.py --image data/xxx/train_dataset/001.png
python inference/preprocess.py --in_channels 3 --img_size 224,224    # 没有 meta 时按 config 备用值（默认 norm=imagenet）
```

检查项：

- 量纲守卫：uint8 的 `0/255` 必须映射到 `(0-mean)/std` 与 `(1-mean)/std`；float 越界（300）/ NaN 必须被拒绝
- blob 的 shape / dtype / 有限性
- `to_blob` 与 `cv2.dnn.blobFromImage` 的排布**逐元素一致**（即 C++ 第 5 步）
- `normalize` 与 `convertTo(1/255) + cv::subtract`/`cv::divide` 写法一致（差 ≤ 1e-5）
- 与训练端 `utils` 的**旧接口**一致（差 ≤ 1e-5，`utils.normalize` 仍按 0~255 量纲调用、内部自动换算）

输出到 `inference/golden/`：

| 文件 | 用途 |
|---|---|
| `golden_input.png` | 无损输入图，C++ 用 `cv::imread` 读它 |
| `golden_resized_u8.npy` | 步骤 1~3 的 uint8 结果，逐步定位 C++ 差异 |
| `golden_blob.npy` | 最终 `1xCxHxW` float32 |
| `golden_spec.json` | 规格与每一步的定义 |

C++ 验收标准：`resized` 逐元素相等；`blob` 最大绝对差 ≤ 1e-5。

## 训练端如何使用

数据增强必须插在**步骤 3（resize）与步骤 4（normalize）之间**，见 `dataset.SegDataset.__getitem__`：

```python
img, mask = self.get_raw(i)        # 步骤 1~3：load_model_image + resize（uint8）
if self.aug:
    img, mask = augment(img, mask, self.aug)     # 增强，输出 float32、0~255 量纲
x = to_blob(normalize(img, self.spec))[0]        # 步骤 4+5 -> CxHxW
```

`SegDataset` 不再接收 `mean/std/img_size`，只接收 `PreprocessSpec`（`train.py` 通过 `cfg["spec"]` 传进来）；
单独调用 `build_datasets` 时由 `dataset.get_spec(cfg)` 依据 `norm/in_channels/img_size` 构造。

训练开始前 `train.check_preprocess_consistency()` 会把「数据集实际喂给网络的张量」与
「同一张原图经 `preprocess()` 的结果」逐元素对一遍，不一致直接报错，避免训练/部署口径不同却静默训完。

`utils.load_image` / `utils.resize_image` / `utils.normalize` 只是对上述实现的薄封装（保留旧签名），
**不要再在 `utils.py` 里加第二份预处理逻辑**；新代码请直接 import `inference.preprocess`。

## 与旧行为的等价性

改造前后对 Dataset 的输出做了逐元素对比（单通道、3 通道单值 mean 广播、非方形 + ImageNet、
labelme JSON 标签、以及**开启数据增强**的情况），x 与 mask 的最大差均为 **0**。
