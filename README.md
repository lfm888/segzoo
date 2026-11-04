# segzoo：基于 segmentation_models.pytorch 的分割模型库

目标：对任意分割数据集，系统地对比 架构 × 编码器，在精度与可部署性（ONNX → RKNN）之间选最合适的设计。

## 目录约定
```
data/<数据集名>/train_dataset/   图像（也可以同时放 labelme 的 .json，当标签用）
data/<数据集名>/labels/          标签：labelme 的 .json，或 mask 图像（按文件名主干匹配）
data/<数据集名>/classes.txt      （可选）类别名，每行一个，可含 background
data/<数据集名>/split/test_train_data.csv （可选）划分文件：image_id,category[,fold]
```
`data_root` 默认是本目录下的 `data`，也可在 `train.py` 的 DEFAULTS 里写绝对路径。

没有 `labels/` 目录时，会自动把 `train_dataset/` 里的 labelme `.json` 当标签用。
类名优先取 `class_names` → `classes.txt` → labelme json 里的标签名（如 inner_circle 的 `center`）→ 默认 `class_1..`。

## 训练/验证划分
| `split_csv` | 行为 |
|---|---|
| `auto`（默认） | 若存在 `<dataset>/split/test_train_data.csv` 就按它划分，否则随机 |
| `none` | 忽略 csv，全部图像按 `seed` + `val_ratio` 随机划分 |
| 具体路径 | 用指定的 csv |

csv 规则（与 `python_model_train` 一致）：`category=train` 参与训练，`category=test` 不参与；
有 `val/valid` 行就直接做验证集；否则含 `fold` 列时 `fold==0` 做验证集；都没有则用 train 子集按 `seed` 随机划分。
`inner_circle_dataset` 已附带与参考项目相同的 `split/test_train_data.csv`（472 train / 52 test）。

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
| `make_demo_data.py` | 生成合成数据（索引 mask / 多通道 mask / labelme json），冒烟测试流程 |
| `train.py` | 训练单个模型，保存 best.pth / 日志 / 可视化 / results.json |
| `sweep.py` | 批量对比 架构×编码器，输出排行榜与帕累托 |
| `export_onnx.py` | best.pth → 静态 shape 的 ONNX（opset 12）并校验 |
| `predict.py` | 无需标签的批量预测可视化（torch 或 onnx 后端） |
| `convert_rknn.py` | ONNX → RKNN（仅 Linux/WSL2，需 rknn-toolkit2） |

## 典型流程
```
python make_demo_data.py
python train.py --dataset _demo_mask --img_size 128,128 --epochs 3        # 冒烟：索引 mask
python train.py --dataset _demo_mask_ch --img_size 128,128 --epochs 3     # 冒烟：多通道 mask
python train.py --dataset inner_circle_dataset --img_size 224,224         # 二值 mask + csv 划分（参考项目同款数据）
python train.py                       # 在 DEFAULTS 指定的数据集上训练
python sweep.py                       # 批量对比
python export_onnx.py && python predict.py --backend onnx
# Linux 上：python convert_rknn.py
```

## 训练输出长什么样
```
==============================================================================
segzoo 训练  |  unet + resnet18 @ 128x128  |  NVIDIA GeForce RTX 4060
==============================================================================
---- [1/4 数据集] --------------------------------------------------------------
  数据集          inner_circle_dataset   (E:\lfm\segzoo\data\inner_circle_dataset)
  样本数          共 472 对  ->  训练 378 / 验证 94   (val_ratio=0.2, seed=42)
  划分方式        csv 的 train 子集按 seed=42 随机划分验证集
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
  产物            best.pth / last.pth / log.csv / results.json / vis/*.png
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
  产物            best.pth  last.pth  log.csv  results.json  vis/*.png
==============================================================================
```
指标说明：`mDice`/`mIoU` 是**前景类平均**（不含背景 0，验证集里不存在的类跳过）；`Acc` 是像素准确率；
每个 epoch 同时打印训练集与验证集指标（便于看是否过拟合）；`log.csv` 记录
`epoch,train_loss,val_loss,train_mdice,train_miou,mdice,miou,pacc,lr,dice_c0...`。
进度条可用 `--progress_bar false` 关闭（非终端环境自动改成每 20% 打一行）。
