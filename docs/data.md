# 数据格式

## 目录结构

```
data/<数据集名>/
├── train_dataset/              原图（png / jpg / bmp / tif）
├── labels/                     标签，与原图按文件名配对
├── classes.txt                 可选：类别名，每行一个，可含 background
├── split/test_train_data.csv   可选：外部划分（image_id,category[,fold]）
├── split.txt                   自动生成：train/val/test 划分，见 split.md
└── label_info.txt              自动生成：标签扫描缓存
```

- **配对规则**：主干名一致即可，支持 `_mask` `_label` `_gt` `_seg` 等后缀。
  例如 `0001.jpg` 可以配 `0001.png`、`0001_label.png`。
- **没有 `labels/`**：会把 `train_dataset/` 里同名的 labelme `.json` 当标签用。
- 数据集路径可以是 `data/xxx`、绝对路径，或直接指向 `train_dataset/`。

## 标签形式（自动识别）

| 标签形式 | 怎么识别 | 类别 |
|---|---|---|
| labelme `.json` | 扩展名 | 每个 `label` 名一类（多边形 / 矩形 / 圆） |
| 单通道索引 png | 单通道，像素值即类别 | 不同像素值 = 不同类，如 `0/1/2` 或 `0/128/255` |
| 多通道 one-hot png | 3 通道，取值 0/255 | 每个通道一类，**允许重叠** |

- 索引图里 `0` 视为背景；没有 `0` 时自动补一个背景类。像素值超过 64 种时按二值 mask 处理（>0 为前景）。
- 多通道 mask 转成训练用的索引图时，**后面的通道覆盖前面的通道**。所以把更具体的类放在后面（例如 Lesion 盖住 Cornea）。
- 全背景或全前景的通道（如 alpha 通道）会被自动忽略。
- 彩色伪彩 mask、三通道完全相同的 mask 会报错，需要先转换格式。
- 强制指定编码：`--mask_mode index` 或 `--mask_mode channels`。

用 `python inspect_data.py` 可以一次查看所有数据集的标签形式、类别、尺寸和像素占比。

## 类别名

优先级从高到低：

1. 训练参数 `class_names`
2. `classes.txt`（数据集根目录或 `labels/` 下）
3. labelme 里出现的标签名（如 `center`）
4. 默认名 `class_1`、`class_2` ……

## 标签扫描缓存 `label_info.txt`

首次训练会解码全部 mask 统计类别，结果写入 `label_info.txt`，之后直接复用。

- 文件里记录了标签目录的指纹（文件数、总字节、最大修改时间）。**标签变了会自动重扫。**
- 改类名只需改 `classes.txt`，不用删缓存（类名的优先级更高）。
- `--label_cache false`：本次强制重扫。
- `--mask_mode` 手动指定时不读写缓存。

用 `python dataset.py --dataset <数据集>` 可以只生成这份缓存和划分，不训练。
