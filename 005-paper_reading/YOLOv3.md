既然代码贴出来了，大家又这么喜欢问，那我就讲讲我对YOLOv3算法的一点点理解吧。万字长文，希望能帮助大家。
翻了下大家开的几百条[issues](https://github.com/YunYang1994/tensorflow-yolov3/issues)，其中的吐槽可以总结成以下三点:

- **YOLOv3算法的前向传播过程怎么进行的，如何理解画网格？** 
- **YOLOv3算法是怎么训练的，损失函数理解太难了，代码写得跟一坨屎一样!** 
- **为什么我在训练的时候loss出现了Nan，有什么办法解决它吗？** 

哈哈，本文的目的，就在于此。

--------------------
# 1. YOLOv3算法的前向传播过程
<p align="center">
    <img width="60%" src="https://pjreddie.com/media/image/Screen_Shot_2018-03-24_at_10.48.42_PM.png" style="max-width:80%;">
    </a>
</p>

假设我们想对上面这张 `416 X 416` 大小的图片进行预测，把图中`dog`、`bicycle`和`car`三种物体框出来，其实涉及到以下三个过程：

- [x] 在图片上找出很多候选框。
- [x] 接着判断候选框里有没有物体？
- [x] 如果有物体的话，那么它属于哪个类别？

听起来就像把大象装进冰箱，分三步走。事实上，目前的 `anchor` 机制算法例如 `RCNN`、`Faster rcnn` 以及 `YOLO` 算法都是这个思想。最早的时候， [`RCNN `](https://arxiv.org/abs/1311.2524)是这么干的，首先利用 `Selective Search`的方法通过图片上像素之间的相似度和纹理特征进行区域合并，然后提出很多候选框并喂给 `CNN` 网络提取特征映射(`feature map`)，最后利用`feature map`训练SVM来对目标和背景进行分类.

![image](https://user-images.githubusercontent.com/30433053/62198083-944b6e00-b3b3-11e9-9cd5-a7230ced3762.png)

这是最早利用神经网络进行目标检测的开山之作，虽然现在看来有不少瑕疵，例如：<br>

- `Selective Search` 会在图片上提取2000个候选区域，每个候选区域都会喂给`CNN`进行特征提取，这个过程太冗余啦，其实这些候选区域之间很多特征其实是可以共享的；<br>
- 由于`CNN`最后一层是全连接层，因此输入图片的尺寸大小也有限制，只能进行Crop或者Warp，这样一来图片就会扭曲、变形和失真；<br>
- 在利用`SVM`分类器对候选框进行分类的时候，每个候选框的特征向量都要保留在磁盘上，很浪费空间！<br>

**尽管如此，但仍不可否认它具有划时代的意义，至少告诉后人我们是可以利用神经网络进行目标检测的。**

后面，一些大神们在此基础上提出了很多改进，从 `Fast RCNN` 到 `Faster RCNN` 再到 `Mask RCNN`, 目标检测的region proposal变得越来越有针对性，提出了著名的 RPN 网络去学习如何画出高质量的候选框，然后再去判断所属物体的类别。简单说来就是: 提供候选框 ---> 然后分类，这就是我们常说的two-stage算法。two-stage 算法的好处就是精度较高，但是检测速度满足不了实时性(real time)的要求。在这样的背景下，YOLO算法应运而生。

## 1.1 不妨先给图片画网格

YOLO算法的最重要的思想就是**画网格**，由于本人做过一点点关于计算流体力学(Computational Fluid Dynamics, 简称CFD)的研究，所以听到网格(grid cells)这个单词感觉特别亲切。emm，先来看看[YOLOv1论文](https://arxiv.org/abs/1506.02640)里的这张图:
<p align="center">
    <img width="70%" src="https://user-images.githubusercontent.com/30433053/62187018-97863000-b39a-11e9-84ff-d7d3166f0407.png" style="max-width:90%;">
    </a>
</p>

初学者咋一看，这特么什么东西？只想说，看不懂！事实上，网上很多关于YOLO系列算法的教程也喜欢拿这张图去忽悠。好了，我想从这张图片出发，讲一讲YOLO算法的**画网格**思想。

### 1.1.1 ROI 映射到 feature map


<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/62198793-e4770000-b3b4-11e9-808f-d53703455def.png" style="max-width:60%;">
    </a>
</p>

### 1.1.2 feature map 长什么样子


| 2维空间|3维空间|
|---|---|
|![image](https://github.com/YunYang1994/SphereFace/blob/master/image/2D_Original_Softmax_Loss_embeddings.gif)|![weibo-logo](https://github.com/YunYang1994/SphereFace/blob/master/image/3D_Original_Softmax_Loss_embeddings.gif)|


### 1.1.3 YOLOv3 的画网格思想





#### 1.2 Darknet53网络提取特征

#### 1.3 很奇怪的Anchor机制

#### 1.4 原来是这样预测的

### 2. YOLOv3的损失函数理解

#### 2.1 边界框损失

#### 2.2 置信度损失

#### 2.3 分类损失

#### 2.4 原来是这样训练的

### 3. YOLOv3的训练技巧

#### 3.1 权重初始化设置

#### 3.2 学习率的设置

#### 3.3 加载预训练模型

#### 3.4 其实好像也没那么难
