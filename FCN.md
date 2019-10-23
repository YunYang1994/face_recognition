## 代码:[TensorFlow2.0-Examples/5-Image_Segmentation/FCN](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/5-Image_Segmentation/FCN)

在我还是实习生的时候，我们组的 leader 讲了 FCN 网络。由于当时对图像分割还不是很了解，所以也没太听懂，只记得他当时讲这篇文章拿了 CVPR-2015 的最佳论文奖。现在学习 FCN 就觉得，这应该是图像分割领域里最经典也最适合入门的网络了吧。

## 网络结构

在我的代码里，使用了 VGG16 作为 backbone 来提取图片特征。如果把 FCN 看成是一个黑箱子，那么我们只要关心网络的输入和输出就行了。如果我们使用 VOC 数据集进行训练，输入图片的维度为 [H,W,C]，那么 FCN 输出的 feature map 形状则为 [H, W, 21]。其中，数字 21 代表的 VOC 的 20 个类别还有 1 个背景。

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/67369222-33df5d80-f5ab-11e9-95d4-3d7813cfa0a8.png" style="max-width:60%;">
    </a>
</p>

FCN 解决的实际问题就是针对图片里的每个像素进行分类，从而完成精确分割。按照以往 CNN 解决分类问题的思路，一般都会在 feature map 后面接一个全连接层，这个全连接层应该有 21 个神经元，每个神经元输出各个类别的概率。但是由于全连接的特征是一个二维的矩阵，因此我们在全连接层之前会使用 Flatten 层将三维的 feature map 展平。这就带来了2个问题：

- 使用了 Flatten 层抹平了图片的空间信息；
- 一旦网络训练好，图片的输入尺寸将无法改变。

FCN 网络很好地解决了这两个问题，它可以接受任意尺寸的输入图像，并保留了原始输入图像中的空间信息，最后直接在 feature map 上对像素进行分类。

## 反卷积层
反卷积操作也称为转置卷积操作(Transposed convolution)。要了解反卷积是怎么回事，得先回顾一下正向卷积的实现过程。假设输入的图片 input 尺寸为 4x4，元素矩阵为:
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
    <img width="30%" src="https://raw.githubusercontent.com/hhaAndroid/conv_arithmetic/master/gif/no_padding_no_strides.gif" style="max-width:30%;">
    </a>
</p>

如果我们用矩阵乘法描述这个过程，那么就是这样子：
![image](https://user-images.githubusercontent.com/30433053/67378150-63956200-f5b9-11e9-8fc7-ef4d1f2ff014.png)



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




