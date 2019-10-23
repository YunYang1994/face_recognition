## 代码:[TensorFlow2.0-Examples/5-Image_Segmentation/Unet](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/5-Image_Segmentation/Unet)

Unet 是 Kaggle 语义分割挑战赛上的常客。因为它简单，高效，易懂，容易定制，最主要的是它可以从相对较小的数据集中学习。在医学图像处理领域，各路高手更是拿着 Unet 各种魔改，既然 Unet 这么强，不妨，先来一段作者的报告热身。

<p align="center">
<video src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-teaser.mp4" width="320" height="180"
controls="controls"></video>
    </a>
</p>

## 网络结构

U-Net 与前面讲到的 [FCN](https://github.com/YunYang1994/ai-notebooks/blob/master/FCN.md)非常的相似，U-Net 比 FCN 稍晚提出来，但都发表在 2015 年，和 FCN 相比，U-Net 的第一个特点是完全对称，也就是左边和右边是很类似的。当我第一次看到该网络的拓扑结构时，顿时惊为天人，简直是一个大写的 **U**。

<p align="center">
    <img width="70%" src="https://user-images.githubusercontent.com/30433053/67412409-cdc9f900-f5f1-11e9-918a-d92355e35395.png" style="max-width:70%;">
    </a>
</p>









