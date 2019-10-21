## RPN 网络
我觉得 RPN 是目标检测领域里最经典也是最容易入门的网络了。如果你想学好目标检测，那一定不能不知道它！今天讲的 RPN 是来一篇来自 CVPR2017 的论文 [Expecting the Unexpected: Training Detectors for Unusual Pedestrians with Adversarial Imposters](https://arxiv.org/pdf/1703.06283)， 作者在 Faster-RCNN 的 RPN 基础上进行了改进，称它为 RPNplus， 用于行人检测。

## 网络结构
下面是 RPN 的网络结构，它采用了 VGG16 网络进行特征提取。从 VGG16 的整体架构来看，作者为了提高 RPN 在不同分辨率图片下的检测率，分别将 Pool3 层、Pool4 层和 Pool5 层的输出进行卷积和融合得到了一个 45 x 60 x 1280 尺寸的 feature map。最后将这个 feature map 分别输入两个卷积层中得到 softmax 分类层与 bboxes 回归层。

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/66986053-a904eb80-f0f0-11e9-93fa-c56fb580f6ae.png" style="max-width:80%;">
    </a>
</p>

## Anchors 机制

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/67201927-fdce9c00-f439-11e9-83cb-dc6aaa7e8c4c.png" style="max-width:80%;">
    </a>
</p>






## 损失函数
为了训练 RPN， 这里采用了 softmax 分类损失函数