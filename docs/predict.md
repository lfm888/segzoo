# 预测与最终评估（`predict.py`）

## 输入目录

`--input_dir` 支持三种情况，代码会自动识别：

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
python predict.py --backend onnx                     # 用导出的 model.onnx 推理，验证前后处理一致
python predict.py --save_vis false                   # 只要指标，不存图（更快）
python predict.py --max_images 8                     # 只跑前 8 张
```

## 输出

- 每张图一张 `<stem>_result.png`：原图 + 半透明分割色块 + 同色虚线边界 + 左上角图例。
- 有标签时，额外输出 `metrics_<split>.json` 并打印指标表。
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

其他样式参数：`--alpha`（色块透明度）、`--thickness`（虚线粗细，默认按图像尺寸）、`--dash` / `--gap`（虚线节奏）、`--legend false`（不画图例）。

## 指标

指标在**模型输入尺寸**上计算：预测结果与最近邻缩小后的真值逐像素比较。
训练、验证、测试使用同一套函数 `utils.seg_metrics`，口径一致。
