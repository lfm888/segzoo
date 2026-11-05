# 固定划分：train / val / test

划分写在 `data/<数据集名>/split.txt`，**第一次生成后就固定复用**，保证不同模型之间可比。

## 三段的用途

| 段 | 谁用 | 做什么 |
|---|---|---|
| train | `train.py` / `sweep.py` | 训练 |
| val | `train.py` / `sweep.py` | 选模型（best.pth）、早停 |
| test | `predict.py` | 模型定型后做**最终泛化评估**；训练脚本不碰它 |

## 文件格式

```
# segzoo 数据划分 train/val/test：自动生成，可手工修改，删除后会重新生成
# 生成时间 2026-11-05 10:32:53 | seed=42
# 来源: test_train_data.csv 的 train 子集按 seed=42 划分 val，csv 的 test 行作为 test
# params: {"seed": 42, "test_ratio": 0.2, "val_ratio": 0.2}
# 共 524 张: train 378 / val 94 / test 52
# 格式: 文件名<TAB>split
0615_new_data_out_0.jpg	train
0615_new_data_out_102.jpg	val
0615_new_data_out_5.jpg	test
```

可以手工修改每一行。

## 生成规则（只在 `split.txt` 不存在时执行）

1. 存在 `split/test_train_data.csv`：沿用它的 `category`。`train` 行抽一部分做 val，`test` 行做 test；有 `fold` 列时 `fold==0` 做 val。
2. 否则按 `seed` 随机打乱，按 `val_ratio` / `test_ratio` 切分（默认值见 `train.py` 顶部的 `DEFAULTS`）。

## 参数变化时会自动重划

`split.txt` 头部记录了生成时用的 `val_ratio` / `test_ratio`。下次运行时如果与当前参数不一致，会**自动重新划分并覆盖**，并打印原因：

```
[data] val_ratio/test_ratio 与 split.txt 生成时不一致（val_ratio: 0.2 -> 0.3），自动重新划分
```

- 只比较真正参与生成的参数。例如 csv 直接给了 test 行时，就不比较 `test_ratio`。
- 旧版本生成、没有 `# params` 行的文件，按默认 0.2 / 0.2 比较。
- `split_check_ratio=false`：永远复用现有文件，不比较参数。
- `resplit=true`：无条件重新生成。

## 新增的图像

不在 `split.txt` 里的图像不会参与训练，会打印警告。要纳入它们，删掉 `split.txt` 重新生成，或手工加一行。

## 只生成划分

```bash
python dataset.py --dataset data/<数据集名>
python dataset.py --dataset data/<数据集名> --val_ratio 0.15 --test_ratio 0.15
```

注意：`dataset.py` 生成的比例要与 `train.py` 的 `val_ratio` / `test_ratio` 保持一致，否则训练时会按 `train.py` 的值重划。
