# segzoo：基于 segmentation_models.pytorch 的分割模型库

目标：对任意分割数据集，系统地对比 架构 × 编码器，在精度与可部署性（ONNX → RKNN）之间选最合适的设计。

## 目录约定
```
data/<数据集名>/train_dataset/   图像（也可以同时放 labelme 的 .json，当标签用）
data/<数据集名>/labels/          标签：labelme 的 .json，或 mask 图像（按文件名主干匹配）
data/<数据集名>/classes.txt      （可选）类别名，每行一个，可含 background
data/<数据集名>/split.txt        固定的 train/val/test 划分（自动生成，可手工改）
data/<数据集名>/label_info.txt   标签扫描缓存（自动生成）
data/<数据集名>/split/test_train_data.csv （可选）外部划分 csv：image_id,category[,fold]
```
`data_root` 默认是本目录下的 `data`，也可在 `train.py` 的 DEFAULTS 里写绝对路径。

## 标签扫描缓存（`label_info.txt`）
第一次训练会解码全部 mask 统计类别（AIDK 768 张约 18s），结果写入 **`data/<数据集名>/label_info.txt`**，
之后每次训练直接复用（毫秒级）。文件是 `key = value`（value 为 JSON）的纯文本，可看可改可删：
```
# segzoo 标签扫描缓存：自动生成，删除后下次训练会重新扫描（手改 classes.txt 优先）
version = 1
generated = 2026-11-05 09:46:08
kind = "mask"
mask_mode = "channels"
num_classes = 4
class_names = ["background", "Cornea", "Iris", "Lesion"]
channel_map = [0, 1, 2]
fingerprint = {"files": 768, "bytes": 10313855, "mtime": 1793152981816731400, "pairs": 768}
```
- 缓存里带了标签目录指纹（文件数/总字节/最大 mtime），**换了标签或重新生成 mask 会自动重扫**
- `classes.txt` / `--class_names` 的优先级高于缓存：改了类名不需要删缓存
- `--mask_mode index|channels` 强制指定编码时会绕过缓存
- `--label_cache false` 强制重新扫描（不读缓存，扫完仍会刷新缓存文件）

没有 `labels/` 目录时，会自动把 `train_dataset/` 里的 labelme `.json` 当标签用。
类名优先取 `class_names` → `classes.txt` → labelme json 里的标签名（如 inner_circle 的 `center`）→ 默认 `class_1..`。

## 固定的三段划分：train / val / test
**一次生成、之后永远复用**，写在 **`data/<数据集名>/split.txt`**（纯文本，可手工改）：
```
# segzoo 数据划分 train/val/test：自动生成，可手工修改，删除后会重新生成
# 生成时间 2026-11-05 10:32:53 | seed=42
# 来源: test_train_data.csv 的 train 子集按 seed=42 划分 val，csv 的 test 行作为 test
# 共 524 张: train 378 / val 94 / test 52（train.py 只用 train+val，predict.py 默认用 test）
# 格式: 文件名<TAB>split
0615_new_data_out_0.jpg	train
0615_new_data_out_102.jpg	val
0615_new_data_out_5.jpg	test
```
| 脚本 | 用到哪一段 |
|---|---|
| `train.py` / `sweep.py` | 只用 **train** 训练 + **val** 选模型/早停；**test 完全不碰** |
| `predict.py` | 默认只在 **test** 上跑，作为最终泛化评估（算 Dice/IoU/P/R/F1） |

生成规则（只在 split.txt 不存在时执行一次）：
1. `<dataset>/split/test_train_data.csv` 存在 → 沿用它的 `category`：`train` 行抽一部分做 val，`test` 行做 test；有 `fold` 列则 `fold==0` 做 val
2. 否则按 `seed` 把全部图像随机分成 **train 60% / val 20% / test 20%**（`--val_ratio` / `--test_ratio` 可改）

相关参数（`train.py`）：
| 参数 | 说明 |
|---|---|
| `split_file` | 划分文件，默认 `split.txt`；设空字符串则不落盘 |
| `resplit` | `true` = 忽略现有 split.txt 重新生成（默认 False，保证不同模型可比） |
| `val_ratio` / `test_ratio` | 只在「生成」时生效（默认 0.2 / 0.2） |
| `split_csv` | 生成时参考的外部 csv：`auto` / `none` / 具体路径 |

重新划分：删掉 `split.txt`（或 `--resplit true`）。新加的图像不在 split.txt 里时会告警并跳过——
想让它们生效就重新生成，或手工加行。

## mask 标签的两种编码（自动识别，可用 `--mask_mode` 强制）
| mask_mode | 适用 | 类别数 |
|---|---|---|
| `index` | 单通道索引图，像素值就是类别值，如 `0/1/2` 或 `0/128/255` | 不同像素值个数（0 为背景；mask 里没有 0 时自动补背景类）；像素值超过 64 种则按二值 mask 处理（0=背景，>0=前景） |
| `channels` | 多通道 mask，每通道 = 1 类，取值 `0/255`，通道间允许重叠（AIDK：R=Cornea、G=Iris、B=Lesion） | 通道数 + 1（背景）；全背景/全前景的通道（如 alpha）自动忽略 |

`channels` 转成训练用的索引图时按「后通道覆盖前通道」，所以更具体的类放在后面的通道（如 Lesion 盖住 Cornea），重叠比例会在日志里打印。
三通道完全相同、或取值不是 0/255 的彩色 mask 会报错提示先转换格式。

## 所有脚本：参数集中在文件顶部 DEFAULTS，直接运行；命令行 --xxx 可覆盖
| 脚本 | 作用 |
|---|---|
| `check_env.py` | 环境自检（依赖、GPU、前向、ONNX 导出） |
| `inspect_data.py` | 探测 `data/` 下所有数据集：标签形式（json / png 单通道索引 / png 多通道 one-hot）、类别、数量、尺寸、磁盘占用 |
| `make_demo_data.py` | 生成合成数据（索引 mask / 多通道 mask / labelme json），冒烟测试流程 |
| `train.py` | 训练单个模型，保存 best.pth / 日志 / 可视化 / results.json |
| `sweep.py` | 批量对比 架构×编码器，输出排行榜与帕累托 |
| `export_onnx.py` | best.pth → 静态 shape 的 ONNX（opset 12）并校验 |
| `predict.py` | 预测 + **在 test 段做最终评估**（Dice/IoU/P/R/F1），每张图一张 `*_result.png`（原图 + 半透明色块 + 同色虚线边界） |
| `convert_rknn.py` | ONNX → RKNN（仅 Linux/WSL2，需 rknn-toolkit2） |

## 典型流程
```
python inspect_data.py                                               # 先看一遍所有数据集的情况
python make_demo_data.py
python train.py --dataset _demo_mask --img_size 128,128 --epochs 3        # 冒烟：索引 mask
python train.py --dataset _demo_mask_ch --img_size 128,128 --epochs 3     # 冒烟：多通道 mask
python train.py --dataset inner_circle_dataset --img_size 224,224         # 二值 mask + csv 划分（参考项目同款数据）
python train.py                       # 在 DEFAULTS 指定的数据集上训练（只看 train/val）
python sweep.py                       # 批量对比（同一份 split.txt，结果可比）
python predict.py                     # 模型定型后：在 test 段上做最终评估 + 可视化
python predict.py --split val         # 想看 val 的效果就用它
python export_onnx.py && python predict.py --backend onnx
# Linux 上：python convert_rknn.py
```

## 数据集探测（`inspect_data.py`）
先跑一遍，确认每个数据集的标签形式与类别数，训练日志里的 `类别数 / 标签编码` 应与它一致。
```
---- [4/6  aidk_fullframe] ----------------------------------------------------
  路径            E:\lfm\segzoo\data\aidk_fullframe
  图像            768 张  .bmp×768
  图像尺寸        1723x1000（抽样 8 张）   3 通道
  标签文件        768 个  .png   9.8 MB
  标签形式        png 多通道 one-hot（每通道=1 类，通道内 >0 为前景）
  标签细节        dtype=uint8  通道数=3
  类别数          4（含背景）
  类别名          0=background  1=Cornea  2=Iris  3=Lesion
  像素占比        background 96.10%  Cornea 2.61%  Iris 0.98%  Lesion 0.31%
...
  数据集                配对/图像   标签形式                    类别  输入尺寸      磁盘
  aidk_fullframe        768/768     png 多通道 one-hot(3ch)     4     1723x1000     3.7 GB
  inner_circle_dataset  524/524     png 单通道二值              2     200x200       9.1 MB
  OCT5K                 3207/3207   png 单通道索引(6 种值)      6     512x512       375.6 MB
```
`--max_scan` 控制每个数据集抽样多少张 mask 做像素/通道统计（`0`=全部），`--data_root` 可换根目录。

## 训练输出长什么样
```
==============================================================================
segzoo 训练  |  unet + resnet18 @ 128x128  |  NVIDIA GeForce RTX 4060
==============================================================================
---- [1/4 数据集] --------------------------------------------------------------
  数据集          inner_circle_dataset   (E:\lfm\segzoo\data\inner_circle_dataset)
  样本数          共 524 对  ->  训练 378 / 验证 94 / 测试 52
  划分方式        split.txt（已存在，固定复用）
  测试集          52 张保留不参与训练/验证，模型定型后用 predict.py 评估（--split test）
  类别数          2（含背景）  0=background  1=center
  标签编码        单通道索引图（像素值 [0, 1]）
  输入            1 通道 @ 128x128   mean=[127.5]  std=[127.5]
---- [2/4 模型] ----------------------------------------------------------------
  结构            unet + resnet18
  参数量          14.322 M / 可训练 14.322 M   (FP32 约 54.6 MB)
  计算量          2.652 GFLOPs (1.326 GMACs)  @ 输入 (1, 1, 128, 128)
---- [3/4 训练配置] ------------------------------------------------------------
  epochs          30  每轮 12 步  共 360 步
  优化器          AdamW  lr=0.001  wd=0.0001  余弦退火
  损失/精度       loss=ce_dice  AMP=True  batch=32  num_workers=0  cache=True
---- [4/4 输出] ----------------------------------------------------------------
  目录            E:\lfm\segzoo\runs\inner_circle_dataset\unet_resnet18_128x128
  产物            best.pth / last.pth / log.csv / results.json / vis/val_XXX.png
==============================================================================

  train [################--------] 9/12  75%  loss 0.1523  0m04s<0m01s   <- 进度条（progress_bar）
------------------------------------------------------------------------------
Epoch   3/30    lr 9.02e-04  |  训练 5s + 验证 1s = 6s  |  已用 17s  剩余 2m14s
  phase  loss      mDice     mIoU      Acc       各类 Dice
  train  0.1523    0.8740    0.7761    0.9923    background 0.997 | center 0.874
  valid  0.1401    0.8902    0.7935    0.9938    background 0.998 | center 0.890   <- best
  valid 前景均值:  Dice 0.8902  IoU 0.7935  P 0.9051  R 0.8758  F1 0.8902
...
==============================================================================
训练完成  |  unet + resnet18 @ 128x128  |  总用时 3m12s
------------------------------------------------------------------------------
  最佳 epoch      3 / 30
  验证指标        mDice(fg) 0.8902  mIoU 0.7935  P 0.9051  R 0.8758  F1 0.8902  Acc 0.9938
  各类 Dice       background 0.998 | center 0.890
  效率            14.322 M 参数  |  2.652 GFLOPs  |  14.34 ms/张 (cuda)
  输出            E:\lfm\segzoo\runs\inner_circle_dataset\unet_resnet18_128x128
  产物            best.pth  last.pth  log.csv  results.json  vis/val_XXX.png
==============================================================================
```
指标说明：`mDice`/`mIoU` 是**前景类平均**（不含背景 0，验证集里不存在的类跳过）；`Acc` 是像素准确率；
每个 epoch 同时打印训练集与验证集指标（便于看是否过拟合）；`log.csv` 记录
`epoch,train_loss,val_loss,train_mdice,train_miou,mdice,miou,pacc,lr,dice_c0...`。
进度条可用 `--progress_bar false` 关闭（非终端环境自动改成每 20% 打一行）。

## 预测 / 最终评估（`predict.py`）
默认只在 **test 段**（`split.txt` 里的 test）上跑，作为模型定型后的泛化评估；有标签时直接算指标并打印表格。
```
python predict.py                            # 最新 best.pth，在 test 段评估 + 出图
python predict.py --split val                # 看 val 段（调试用）
python predict.py --split all --metrics false # 没标签的新图：全部推理，只要图
python predict.py --max_images 8             # 只跑 8 张（默认 0 = 全部）
python predict.py --backend onnx             # 用导出的 model.onnx 评估，顺便验证前后处理一致性
python predict.py --save_vis false           # 只要指标，不存图（快很多）
```
输出：
- 每张图一张 `<stem>_result.png`（原图 + 半透明分割色块 + 同色虚线边界 + 图例）
- `<out_dir>/metrics_test.json`（含每类 dice/iou/P/R/F1、前景均值、像素准确率、ckpt 路径等）
```
==============================================================================
最终评估（test 段）| unet_resnet18_128x128 | unet + resnet18 @ 128x128 | 统计 52 张
------------------------------------------------------------------------------
  cls  name           Dice     IoU      P        R        F1
  0    background     0.9821   0.9648   0.9803   0.9839   0.9821
  1    center         0.9432   0.8925   0.9510   0.9356   0.9432
------------------------------------------------------------------------------
  前景均值    Dice 0.9432  IoU 0.8925  P 0.9510  R 0.9356  F1 0.9432   像素准确率 0.9774
==============================================================================
```
指标在**模型输入尺寸**上算（预测结果与最近邻缩小的 GT 逐像素比），与训练/验证时用同一套 `utils.seg_metrics`。

## 预测结果可视化细节
- 底图 = 原图（原分辨率）
- 每个类别 = **半透明色块** + **同色虚线边界**（边界画在色块之上，线宽/虚线节奏按图像尺寸自动取）
- 左上角图例：当前图出现的类别（颜色与虚线/色块一致，类名取自训练时的 `meta.class_names`）

```
python predict.py --alpha 0.3 --thickness 2        # 色块更淡 / 虚线更粗（默认 = 短边/600，即 1~2px）
python predict.py --thickness 1                    # 强制 1px 虚线（不抗锯齿，虚线节奏更清楚）
python predict.py --legend false                   # 不画图例
python predict.py --dash 12 --gap 8                # 手动指定虚线节奏（默认按线宽自动）
```
调色板来自 `utils.PALETTE`（下标=类别索引，0=背景不画），类别数超过 8 时自动用 HSV 补色。
虚线是用弧长重采样画的（OpenCV 没有原生虚线），所以任意形状的轮廓都是均匀虚线。

## 验证集可视化（`train.py` 的 `vis/`）
训练结束后按 `vis_samples` 抽验证集样本，每张**只存一张** `vis/val_007.png`，一图同时给出 GT 与预测：

| 元素 | 含义 |
|---|---|
| 底图 | 原图（**原分辨率**，预测结果最近邻放大回原图尺寸后再叠加，和 `predict.py` 一致） |
| **半透明色块** | **预测结果**（按类别上色） |
| **同色虚线边界** | **真值 mask**（GT） |
| 左上图例 | 当前图出现的类别（GT 或预测里出现即列出） |

所以「色块与虚线重合」= 预测正确；虚线露出色块外 = 漏检；只有色块没有虚线 = 误检。
渲染与 `predict.py` 共用 `utils.render_prediction(..., edge_mask=gt)`；
色块透明度用 `--vis_alpha`（默认 0.45，同 `predict.py --alpha`）。
