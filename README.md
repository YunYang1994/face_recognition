## 人脸识别
--------------------
这个仓库是使用`TensorFlow 2.0`框架，并基于 [cvpr2019-arcface](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf) 论文上完成的，其中主要分为四大块：人脸检测、人脸矫正、提取特征和特征比对。各个模块的大小和在我的 17 款 macbook-pro 的 CPU 上跑耗时如下：

- 人脸检测：使用的是 mtcnn 网络，模型大小约 1.9MB，耗时约 30ms；
- 人脸矫正：OpenCV 的仿射变换，耗时约 0.83ms；
- 提取特征：使用 MobileFaceNet（或IResNet）网络，耗时约30ms；
- 特征比对：使用曼哈顿距离，单次搜索和完成比对耗时约 0.011 ms；

<p align="center">
    <img width="80%" src="https://raw.githubusercontent.com/YunYang1994/face_recognition/master/weights/demo.png" style="max-width:65%;">
    </a>
</p>

## 注册人脸
--------------------

注册人脸的方式有两种，分别是:

1. 打开相机注册:

```bashrc
$ python register_face.py -person Sam -camera
```

按 `s` 键保存图片，需要在不同距离和角度拍摄 10 张图片或者按 `q` 退出。

2. 导入人脸图片:

保证文件的名字与注册人名相同，并且每张图片只能出现一张这个 ID 的人脸。


```bashrc
$ python register_face.py -person Jay
```

## 识别人脸
--------------------

|Method | LFW(%) | CFP-FP(%) | AgeDB-30(%) | MegaFace(%)| cpu-time | weights |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MobileFaceNet | 99.50 | 88.94 | 95.91 | --- | 35ms | [下载链接](https://github.com/YunYang1994/face_recognition/blob/master/weights/mobilefacenet.npy)
| IResNet | 99.77 | 98.27 | 98.28 | 98.47 | 435ms | [提取码: xgmo](https://pan.baidu.com/s/1QIYpHYazaPMTI0E15WRGug)

识别模型用的是 `MobileFaceNet` 网络，这里直接使用了 [insightface](https://github.com/deepinsight/insightface) 在 ms1m-refine-v1 三百万多万张人脸数据集上训练的模型。这部分工作在 `mxnet` 分支上，你可以通过 `git checkout mxnet` 进行切换。

由于该模型是 mxnet 格式，因此使用了 [mmdnn](https://github.com/microsoft/MMdnn) 导出了其模型权重 `mobilefacenet.npy`。接着使用了 `TF2` 自己手写了一个 `MobileFaceNet` 网络并导入权重，预测精度没有任何损失。这部分工作在 `master` 分支上。

最后，如果你要识别人脸，可以执行：

```bashrc
$ python main.py
```


