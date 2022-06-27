# 可微渲染新视角生成赛题：multistage NeRF

## 项目介绍

TODO

## 算法介绍

### NeRF

TODO

### instant ngp(JNeRF)

TODO

### realESRGAN

TODO

### VGG19（ImageNet预训练）

> 用于realESRGAN模型

TODO

### 算法流程（优化后）

1. NeRF生成训练集上的低质量采样图片

2. 使用缩放、裁剪进行数据增强

3. 使用训练集上的图片训练增强渲染模型

4. NeRF+增强渲染模型两阶段训练&生成测试集上结果

### 创新点

1. multistage nerf

    TODO：描述两阶段训练&生成：先训练NeRF，再训练增强渲染模型，最后两级生成（可以画个简图）

2. scene-aware modeling

    TODO：说明每个场景的做法不一样（Car：优化后的算法，Easyship：NeRF，其他：JNeRF）

3. 对于不同NeRF模型的适配

    TODO：表达“增强渲染模型是通用的”这个意思

4. 仿射变换、缩放裁剪数据增强用于NeRF以及增强渲染模型

    TODO：说明仅使用了给的训练集&验证集，和数据增强的好处（泛泛）

### 效果展示及实验结果

## 安装

### 运行环境

+ 单卡GPU：GeForce RTX 3090
+ Linux系统

### 安装依赖

+ CUDA环境

+ Python第三方库

    ```bash
    pip install -r ./requirements.txt
    ```

## 训练&推理

单卡训练可运行以下命令：

```bash
# Easyship
bash train_Easyship.sh

# Car
bash train_Car.sh

# Else
bash train_Else.sh
```

## 致谢

```txt
@article{hu2020jittor,
    title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
    author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
    journal={Science China Information Sciences},
    volume={63},
    number={222103},
    pages={1--21},
    year={2020}
}
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
@inproceedings{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    booktitle={ECCV},
}
@Article{wang2021realesrgan,
    title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    journal={arXiv:2107.10833},
    year={2021}
}
```

> 部分代码参考自[jittor-GAN](https://github.com/Jittor/gan-jittor)开源模型库
