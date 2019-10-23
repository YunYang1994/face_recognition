# 代码：[TensorFlow2.0-Examples/4-Object_Detection/RPN](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/RPN)
## RPN 网络（区域生成网络）
我觉得 RPN 是目标检测领域里最经典也是最容易入门的网络了。如果你想学好目标检测，那一定不能不知道它！今天讲的 RPN 是来一篇来自 CVPR2017 的论文 [Expecting the Unexpected: Training Detectors for Unusual Pedestrians with Adversarial Imposters](https://arxiv.org/pdf/1703.06283)， 作者在 Faster-RCNN 的 RPN 基础上进行了改进，用于行人检测。

## 网络结构
下面是 RPN 的网络结构，它采用了 VGG16 网络进行特征提取。从 VGG16 的整体架构来看，作者为了提高 RPN 在不同分辨率图片下的检测率，分别将 Pool3 层、Pool4 层和 Pool5 层的输出进行卷积和融合得到了一个 45 x 60 x 1280 尺寸的 feature map。最后将这个 feature map 分别输入两个卷积层中得到 softmax 分类层与 bboxes 回归层。

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/66986053-a904eb80-f0f0-11e9-93fa-c56fb580f6ae.png" style="max-width:80%;">
    </a>
</p>

## Anchor 机制
目标检测其实是生产很多框，然后在消灭无效框的过程。生产很多框的过程利用的是 Anchor 机制，消灭无效框则采用非极大值抑制过程进行处理。RPN 网络输入的图片为 720 x 960，输出的 feature map 尺寸为 45 x 60，由于它们每个点上会产生 9 个 anchor boxes，因此最终一共会得到 45 x 60 x 9 个 anchor boxes。

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/67201927-fdce9c00-f439-11e9-83cb-dc6aaa7e8c4c.png" style="max-width:80%;">
    </a>
</p>

直接利用这些 anchor boxes 对真实框进行预测会有些困难，因此作者采用了 anchor boxes 与 gt-boxes 的偏移量机制进行回归预测。

<p align="center">
    <img width="50%" src="https://user-images.githubusercontent.com/30433053/67203842-b4348000-f43e-11e9-812a-8a1f56f5e125.png" style="max-width:50%;">
    </a>
</p>

```python
def compute_regression(box1, box2):
    """
    box1: ground-truth boxes
    box2: anchor boxes
    """
    target_reg = np.zeros(shape=[4,])
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    target_reg[0] = (box1[0] - box2[0]) / w2
    target_reg[1] = (box1[1] - box2[1]) / h2
    target_reg[2] = np.log(w1 / w2)
    target_reg[3] = np.log(h1 / h2)

    return target_reg
```
## 损失函数
RPN 的损失函数和 YOLO 非常像，不过从发表论文时间顺序来看，应该是 YOLO 借鉴了 RPN 。在 Faster-rcnn 论文里，RPN 的损失函数是这样的:

- 为了训练 RPN， 我们首先给每个 anchor boxes 设置了两个标签，分别为 0: 背景, 1: 前景；
- 与 ground-truth boxes 重合度 (iou) 最高的那个 anchor boxes 设置为正样本;
- 只要这个 anchor boxes 与任何一个 ground-truth boxes 的 iou 大于 0.7，那么它也是一个正样本；
- 如果 anchor boxes 与所有的 ground-truth boxes 的 iou 都小于 0.3， 那么它就是一个负样本，表示不包含物体；
- 在前面这几种情况下，已经能够产生足够多的正、负样本了，剩下的则既不是正样本，也不是负样本，它们不会参与到 RPN 的 loss 的计算中去。

在我的代码 [demo.py](https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/4-Object_Detection/RPN/demo.py) 里将正负样本都可视化出来了，大家只要配置好 image 和 label 的路径然后直接执行 `python demo.py` 就可以看到以下图片。

<p align="center">
    <img width="80%" src="https://user-images.githubusercontent.com/30433053/67204319-db3f8180-f43f-11e9-99fe-bb73b0123fc6.png" style="max-width:80%;">
    </a>
</p>

在上图中，蓝色框为 anchor boxes，它们就是正样本，红点为这些正样本 anchor boxes 的中心位置，黑点表示的是负样本 anchor boxes 的中心位置。从图中可以看出：在有人的区域，正样本框的分布比较密集，并且红点都在人体中心区域；而在没有人的区域则布满了黑点,它们表示的是负样本，都属于背景。

在前面讲到，RPN 网络预测的是 anchor boxes 与 ground-truth boxes 的偏移量，那如果我们将这些正样本 anchor boxes 的偏移量映射回去的话：

```
=> Decoding positive sample: 20, 20, 0
=> Decoding positive sample: 20, 20, 7
...
=> Decoding positive sample: 36, 31, 1
```

<p align="center">
    <img width="80%" src="https://user-images.githubusercontent.com/30433053/67206752-e052ff80-f444-11e9-94eb-f3ce7caacfd3.png" style="max-width:80%;">
    </a>
</p>

你会发现，这就是 ground-truth boxes 框（绿色框）和物体中心点（红色点）的位置。事实上，RPN 的损失是一个多任务的 loss function，集合了分类损失与回归框损失，它们两者之间的优化可以通过 λ 系数去实现平衡。

<p align="center">
    <img width="40%" src="https://user-images.githubusercontent.com/30433053/67206340-07f59800-f444-11e9-9126-5484ea68cdd3.png" style="max-width:40%;">
    </a>
</p>

初次看这个损失函数有点迷，它其实是一个 smooth-L1 损失函数，如下图所示。并且，正负样本都会参与到分类损失的反向传播中去（因为你需要告诉网络什么是正样本和负样本），而回归框的损失只有正样本参与计算（只有正样本才有回归框损失，负样本作为背景是没有回归框损失的)。

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/67206291-f4e2c800-f443-11e9-8a15-a41e394f2c21.png" style="max-width:60%;">
    </a>
</p>

```python
def compute_loss(target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes):
    """
    target_scores shape: [1, 45, 60, 9, 2],  pred_scores shape: [1, 45, 60, 9, 2]
    target_bboxes shape: [1, 45, 60, 9, 4],  pred_bboxes shape: [1, 45, 60, 9, 4]
    target_masks  shape: [1, 45, 60, 9]
    """
    score_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_scores, logits=pred_scores)
    foreground_background_mask = (np.abs(target_masks) == 1).astype(np.int)
    score_loss = tf.reduce_sum(score_loss * foreground_background_mask, axis=[1,2,3]) / np.sum(foreground_background_mask)
    score_loss = tf.reduce_mean(score_loss)

    boxes_loss = tf.abs(target_bboxes - pred_bboxes)
    boxes_loss = 0.5 * tf.pow(boxes_loss, 2) * tf.cast(boxes_loss<1, tf.float32) + (boxes_loss - 0.5) * tf.cast(boxes_loss >=1, tf.float32)
    boxes_loss = tf.reduce_sum(boxes_loss, axis=-1)
    foreground_mask = (target_masks > 0).astype(np.float32)
    boxes_loss = tf.reduce_sum(boxes_loss * foreground_mask, axis=[1,2,3]) / np.sum(foreground_mask)
    boxes_loss = tf.reduce_mean(boxes_loss)

    return score_loss, boxes_loss
```
## k-means 造框

如果 Anchor boxes 的尺寸选得好，那么就使得网络更容易去学习。刚开始我以为反正网络预测的都是 Bounding Boxes 的偏移量，那么 Anchor boxes 尺寸就没那么重要了。但我在复现算法和写代码的过程中发现，看来我还是太年轻了。我使用的是 [synthetic_dataset 数据集](https://pan.baidu.com/s/1QZAIakMVS0sJV0sjgv7v2w&shfl=sharepset)进行训练，该数据集里所有检测的目标都为 "person"，假如我直接用作者[论文](https://arxiv.org/pdf/1703.06283)里的原始 anchor，那么得到的正样本为如下左图；而如果我使用 [k-means](https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/4-Object_Detection/RPN/kmeans.py)算法对该数据集所有的 ground-truth boxes 进行聚类得到的 anchor，那么效果就如下右图所示，显然后者的效果比前者好得多。

| 论文原始 anchor | k-means 的 anchor|
|---|---
|![image](https://user-images.githubusercontent.com/30433053/67209048-3a55c400-f449-11e9-944f-1efd2029d408.png)|![image](https://user-images.githubusercontent.com/30433053/67209522-1ba3fd00-f44a-11e9-9c9b-d2c14f0d6014.png)

不仅如此，事实上一些其他超参数也会影响正负样本的分布情况，从而直接影响到网络的学习过程。所有这些事实都告诉我们，学习神经网络不能靠从网上看一些浅显的教程就够了的，关键还得自己去多多看源码并实践，才能成为一名合格的深度学习炼丹师。

| pos_thresh=0.2, neg_thresh=0.1 | pos_thresh=0.7, neg_thresh=0.2|
|---|---
|![image](https://user-images.githubusercontent.com/30433053/67210062-08456180-f44b-11e9-9719-2bf8cb10ac74.png)|![image](https://user-images.githubusercontent.com/30433053/67210282-66724480-f44b-11e9-8cf5-fa9372131555.png)

最后在测试集上的效果，还是非常赞的! 训练的 score loss基本降到了零，boxes loss 也是非常非常低。但是由于是 RPN 网络，所以我们又不能对它抱太大期望，不然 Faster-RCNN 后面更精确的回归层和分类层意义就不大了。

|![image](https://user-images.githubusercontent.com/30433053/67265442-5789a180-f4e0-11e9-9fcd-6e72136c2913.png)|![image](https://user-images.githubusercontent.com/30433053/67265549-915aa800-f4e0-11e9-91e8-87ee05b7748c.png)|
|---|---
|![image](https://user-images.githubusercontent.com/30433053/67265487-6c663500-f4e0-11e9-8bf9-f9d59d22b0a8.png)|![image](https://user-images.githubusercontent.com/30433053/67265620-c36c0a00-f4e0-11e9-8689-9d3b6efaff47.png)

[【推荐: YOLOv3 的算法复现笔记, TensorFlow2.0-Examples/4-Object_Detection/YOLOv3】](https://github.com/YunYang1994/ai-notebooks/blob/master/YOLOv3.md)
