## 代码:[TensorFlow2.0-Examples/5-Image_Segmentation/FCN](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/5-Image_Segmentation/FCN)

在我还是实习生的时候，我们组的 leader 讲了 FCN 网络。由于当时对图像分割还不是很了解，所以也没太听懂，只记得他当时讲这篇文章拿了 CVPR-2015 的最佳论文奖。现在学习 FCN 就觉得，这应该是图像分割领域里最经典也是最适合入门的网络了吧。

<p align="center">
    <img width="50%" src="https://user-images.githubusercontent.com/30433053/67369222-33df5d80-f5ab-11e9-95d4-3d7813cfa0a8.png" style="max-width:50%;">
    </a>
</p>

## 网络结构

在我的代码里，使用了 VGG16 作为 backbone 来提取图片特征【其实作者也使用了 VGG19 作为backbone，但是发现效果和 VGG16 差不多。】。如果把 FCN 看成是一个黑箱子，那么我们只要关心网络的输入和输出就行了。如果我们使用 VOC 数据集进行训练，输入图片的维度为 [H,W,C]，那么 FCN 输出的 feature map 形状则为 [H, W, 21]。其中，数字 21 代表的 VOC 的 20 个类别还有 1 个背景。

<p align="center">
    <img width="80%" src="https://user-images.githubusercontent.com/30433053/67388473-4cf80680-f5cb-11e9-9c36-36480d84b48d.png" style="max-width:80%;">
    </a>
</p>

FCN 解决的实际问题就是针对图片里的每个像素进行分类，从而完成精确分割。按照以往 CNN 解决分类问题的思路，一般都会在 feature map 后面接一个全连接层，这个全连接层应该有 21 个神经元，每个神经元输出各个类别的概率。但是由于全连接的特征是一个二维的矩阵，因此我们在全连接层之前会使用 Flatten 层将三维的 feature map 展平。这就带来了2个问题：

- 使用了 Flatten 层抹平了图片的空间信息；
- 一旦网络训练好，图片的输入尺寸将无法改变。

FCN 网络很好地解决了这两个问题，它可以接受任意尺寸的输入图像，并保留了原始输入图像中的空间信息，最后直接在 feature map 上对像素进行分类。

## 跳跃拼接
在刚开始的时候，作者将输入图片经过卷积和下采样操作一头走到尾，最后宽和高都被缩放了 32 倍。为了将 feature map 上采样到原来的尺寸，因此作者将 vgg16 的输出扩大了 32 倍，并将该模型称为 FCN-32s。

![image](https://user-images.githubusercontent.com/30433053/67386859-53d14a00-f5c8-11e9-9d62-ccb1c2e61a80.jpg)
但是发现FCN-32s的分割效果并不够好，如下图所示。尽管最后的 feature map 输出经过了 32 倍的上采样操作，但是图片里的边缘细节信息还是被 VGG16 网络里的卷积和下采样操作所模糊掉了。

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/67385904-9003ab00-f5c6-11e9-87da-3dbf0dcb079a.png" style="max-width:60%;">
    </a>
</p>

作者把它称作是一个**what**和**where**的问题，请看下面作者的原话：

>Semantic segmentation faces an inherent tension between semantics and location: global information resolves what while local information resolves where.

说白了就是**全局信息能够预测这个物体是哪个类别，而局部的细粒度信息能够实现对物体的定位与检测**。为了解决这个问题，作者通过缓慢地（分阶段地）对编码特征进行上采样，从浅层添加了“skip connections(跳跃连接)”，并将这两个特征映射相加，并最终将它上采样 8 或者 16 倍进行输出，分别称为 FCN-8s 和 FCN-16s 模型。

![image](https://user-images.githubusercontent.com/30433053/67389318-f4c20400-f5cc-11e9-9769-acb912aa8292.png)

添加 skip connections 结构后，就能将深层的，粗糙的语义信息与浅层的，精细的表面信息融合起来，从而在一定程度上解决图像边缘分割效果较差的问题。

>We define a skip architecture to take advantage of this feature spectrum that combines deep, coarse, semantic information and shallow, fine, appearance information

## 反卷积层
FCN的上采样层使用的是反卷积层，反卷积也称为转置卷积操作(Transposed convolution)。要了解反卷积是怎么回事，得先回顾一下正向卷积的实现过程。假设输入的图片 input 尺寸为 4x4，元素矩阵为:
<p align="center">
    <img width="40%" src="https://user-images.githubusercontent.com/30433053/67377252-f6350180-f5b7-11e9-8e27-50f2db5d5bae.png" style="max-width:40%;">
    </a>
</p>

卷积核的尺寸为 3x3，其元素矩阵为：
<p align="center">
    <img width="25%" src="https://user-images.githubusercontent.com/30433053/67377437-3b593380-f5b8-11e9-8c01-f3e5e884e173.png" style="max-width:25%;">
    </a>
</p>

正向卷积操作：步长 strides = 1, 填充 padding = 0,输出形状为 2x2，该过程如下图所示：
<p align="center">
    <img width="25%" src="https://raw.githubusercontent.com/hhaAndroid/conv_arithmetic/master/gif/no_padding_no_strides.gif" style="max-width:30%;">
    </a>
</p>

在上面这幅图中，底端为输入，上端为输出，卷积核为 3x3。如果我们用矩阵乘法去描述这个过程：
<p align=left">
    <img width="80%" src="https://user-images.githubusercontent.com/30433053/67378378-bbcc6400-f5b9-11e9-8d80-672010380f1c.png" style="max-width:80%;">
    </a>
</p>
稀疏矩阵 C 的形状为 4x16, X 形状为 16x1，Y 的形状为 4x1，将 Y 进行 reshape 后便是我们的期望输出形状 2x2。那么，反卷积的操作就是要对这个矩阵运算过程进行转置，通过输出 Y 得到输入 X：

<p align="center">
    <img width="15%" src="https://user-images.githubusercontent.com/30433053/67379139-eff45480-f5ba-11e9-99bc-9fcbdc731290.png" style="max-width:15%;">
    </a>
</p>

从矩阵元素形状的角度出发，可以理解为：**16x1=16x4x4x1**，下面这个动画比较生动地描述了反卷积过程:

<p align="center">
    <img width="25%" src="https://raw.githubusercontent.com/hhaAndroid/conv_arithmetic/master/gif/no_padding_no_strides_transposed.gif" style="max-width:30%;">
    </a>
</p>

值得注意的是，反卷积操作并不是卷积操作的可逆过程，也就是说图像经过卷积操作后是不能通过反卷积操作恢复原来的样子。这是因为反卷积只是转置运算，并非可逆运算。

## 数据处理
在 PASCAL VOC 数据集中，每个类别对应一个色彩【RGB】, 因此我们需要对`SegmentationClass`文件夹里的每张 mask 图片根据像素的色彩来标定其类别，在代码 [parser_voc.py](https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/5-Image_Segmentation/FCN/parser_voc.py#L48)是这样进行处理的。

```python
for i in range(H):
   write_line = []
   for j in range(W):
   	pixel_color = label_image[i, j].tolist() # 得到该像素点的 RGB 值
        if pixel_color in colormap:
       	    cls_idx = colormap.index(pixel_color) # 索引该 RGB 值的类别
        else:
            cls_idx = 0
```
|![image](https://user-images.githubusercontent.com/30433053/66732790-d4d56680-ee8f-11e9-9120-07b0e8aa53d4.jpg)|![image](https://user-images.githubusercontent.com/30433053/66732791-d69f2a00-ee8f-11e9-9c5d-16cc84bc7e9e.jpg)|![image](https://user-images.githubusercontent.com/30433053/66732795-da32b100-ee8f-11e9-9d85-f0ddba7a3ab1.jpg)|
|---|---|:---:|
|![image](https://user-images.githubusercontent.com/30433053/66732799-dd2da180-ee8f-11e9-9025-3a3e0e94a20b.jpg)|![image](https://user-images.githubusercontent.com/30433053/66733895-aa85a800-ee93-11e9-8eae-405235aa8564.jpg)|![image](https://user-images.githubusercontent.com/30433053/66733897-ace80200-ee93-11e9-84e4-21f7d94d06eb.jpg)|

考虑到在批量训练图片时的 batch_size >= 1，因此必须将图片 resize 成相同的尺寸，这里采用的是最近邻插值法，从而保证新插值的像素分类问题。

## 模型训练

如果你要训练 FCN-8s 的话，还是推荐你加载 VGG16 模型的，否则会变得非常耗时。还有一点的就是，其实训练图片里的像素类别是非常不均衡的。例如 75% 的图片像素都属于背景（见上图），因此你会发现在训练时其精度很快就达到了80%，但此时的预测结果却是一片黑，即预测的类别都为背景。

对于这种情况，学术界有两种办法： Patchwise training 和类别损失加权的方法来进行训练。

- Patchwise training: 补定式训练方法，它旨在避免全图像训练的冗余。在语义分割中，由于要对图像中的每个像素进行分类，如果输入整个图像可能会有大量的冗余。因此在训练分割网络时，避免这种情况的一种标准方法是从训练集而不是完整图像中给网络提供成批的随机补丁（感兴趣对象周围的小图像区域）。从另一种角度出发，我们也可以使得这些补丁图片尽量减少背景信息，从而缓解类别不均衡问题。

- 类别损失加权: 根据类别数量的分布比例对各自的损失函数进行加权，比如有些样本的数量较少，我就给它的损失函数比重增大一些。

对此，作者根据实验结果放话了：

>We explore training with sampling in Section 4.3, and do not find that it yields faster or better convergence for dense prediction. Whole image training is effective and efficient.

补丁式训练完全没有必要，训练 FCN 还是输入整张图片比较好。并且解决这种类别不均衡的问题，只需要给损失函数按比例加权重就行。最后作者还对此进行了学术上的解释，我这里就不讲了，话讲多了你们会觉得我在胡言乱语...





