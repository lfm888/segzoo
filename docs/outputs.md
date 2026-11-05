# 训练输出与指标

## 输出目录

```
runs/<数据集名>/<arch>_<encoder>_<H>x<W>/
├── best.pth        验证集 mDice 最高的权重（部署/预测用这个）
├── last.pth        最后一轮的权重
├── log.csv         每个 epoch 的全部指标
├── results.json    最佳结果汇总（指标、参数量、GFLOPs、延迟、curves 路径）
├── curves.png      训练曲线二联图
└── vis/            验证集可视化：val_XXX.png（虚线=真值，色块=预测）
```

`sweep.py` 会在 `runs/<数据集名>/` 下额外生成 `sweep_results.csv`（排行榜与帕累托标记）。

## 控制台输出

训练开始时打印四段配置（数据集 / 模型 / 训练配置 / 输出），每个 epoch 打印一张小表：

```
Epoch   3/30    lr 9.02e-04  |  训练 5s + 验证 1s = 6s  |  已用 17s  剩余 2m14s
  phase  loss      mDice     mIoU      Acc(fg)   各类 Dice
  train  0.1523    0.8740    0.7761    0.9521    background 0.997 | center 0.874
  valid  0.1401    0.8902    0.7935    0.9634    background 0.998 | center 0.890   <- best
  valid 前景均值:  Dice 0.8902  IoU 0.7935  P 0.9051  R 0.8758  F1 0.8902
  valid 准确率:    全部像素 0.9938  前景像素 0.9634（前景占 8.42%）
```

同时打印训练集和验证集，便于判断过拟合。

## 指标含义

- **mDice / mIoU / P / R / F1**：前景类的平均值，不含背景。验证集里不存在的类会被跳过。
- **准确率有两种口径**，两种都会计算和保存：
  - `全部像素`：包含背景。前景占比小的数据集上很容易接近 1，参考价值有限。
  - `前景像素`（`Acc(fg)`）：只统计真值为前景的像素。
- 参数 `acc_exclude_bg`（默认 `true`）决定表格和汇总里的主指标用哪种口径。

## log.csv 列

```
epoch, train_loss, val_loss, train_mdice, train_miou, train_pacc, train_acc_fg,
mdice, miou, pacc, acc_fg, lr, dice_c0, dice_c1, ...
```

`results.json` 中的 `fg_ratio` 是前景像素占比。

## curves.png

左图：train / valid 的 loss（左轴）和准确率（右轴），两轴范围都固定为 0~1。
右图：验证集每类 Dice。`acc_exclude_bg=true` 时不画背景。
黑色虚线标出 `best_epoch`。

超出 0~1 的 loss 不会画出，图的左下角会注明真实数值，`log.csv` 中也保留原值。`--curve_ylim 0` 可改为自动范围。

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `acc_exclude_bg` | `true` | 主准确率只统计前景像素 |
| `curve_plot` | `true` | 训练结束画 `curves.png` |
| `curve_ylim` | `1.0` | 曲线 y 轴上限，0 为自动 |
| `label_smoothing` | `0.0` | CE 的 label smoothing，建议 0.05 |
| `early_stop_patience` | `20` | 验证 mDice 连续多少轮无提升则停止，0 为关闭 |
| `vis_samples` | `8` | 训练结束保存多少张验证集可视化 |
| `progress_bar` | `true` | 进度条；非终端环境自动改为每 20% 输出一行 |

参数的完整列表见 `train.py` 顶部的 `DEFAULTS`。
