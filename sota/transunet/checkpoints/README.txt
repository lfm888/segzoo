把 TransUNet 的 R50-ViT-B/16 预训练权重放在这个目录下，文件名必须是：

    R50+ViT-B_16.npz        （约 380MB）

下载地址（官方 Google Drive）：
    https://github.com/Beckschen/TransUNet  的 README「Getting Started / Download Google pre-trained ViT models」
    或 https://console.cloud.google.com/storage/browser/vit_models/imagenet21k

放好后无需改代码：--encoder_weights imagenet（默认）会自动加载并调用 model.load_from()。
没有这个文件也能训练，模型从零初始化（105.2M 参数），小数据集上效果会明显差。

注意：权重文件不要提交到 git（见 .gitignore）。
