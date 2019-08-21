看了很多关于反卷积的文章，只想说它们写的都是一坨翔。关于反卷积操作，大家不要想得那么复杂，说白了其实就是**卷积操作的一个逆向过程!** 反正牢记这一点就够了，在这里我不想贴一大堆任何人都看不懂的公式和矩阵理论，只想贴出最简洁的代码帮你了解什么是反卷积操作。

# 卷积操作
假设有一个通道数为 1 的 feature map，它的宽和高都为 4，我们用一个 3x3 的卷积核对它进行卷积操作 (strides=1, padding="VALID")，最后我们便得到了一个宽和高都为 2 的 feature map。这个过程如下:

<p align="center">
    <img width="25%" src="https://user-images.githubusercontent.com/30433053/63404840-4dcbbb00-c417-11e9-8d35-0eea90c5a3c6.gif" style="max-width:25%;">
    </a>
</p>

>在上图中，输入为 4x4, 经卷积操作后得到 2x2 的形状。

```python
import tensorflow as tf

x = tf.constant(1., shape=[1,4,4,1])
w = tf.constant(1., shape=[3,3,1,1]) # [卷积核的高度，卷积核的宽度，输入通道数，输出通道数]
y = tf.nn.conv2d(x, w, strides=[1,1,1,1,], padding='VALID')
print(y.shape) # [1, 2, 2, 1]
```

# 反卷积操作

![image](https://user-images.githubusercontent.com/30433053/63409704-45c64800-c424-11e9-9f61-c78b5f27c51c.png)
这里只解释五个非常重要的参数，它们分别是：
- value: 输入张量，shape = [batch, height, width, in_channels]
- filter: 卷积核， shape = [height, width, output_channels, in_channels]
- output_shape： 反卷积操作的输出形状
- strides： 一个整数列表,输入张量的每个维度的滑动窗口的步幅
- padding：一个字符串,'VALID' 或者 'SAME'

好了，我们现在要利用反卷积操作对上面的卷积过程进行逆转过来。在这个反卷积操作过程中，卷积核的形状保持不变，并且 strides 和 padding 的操作依然不会发生变化，**唯一不同的是卷积过程里的输出在反卷积过程里变成了输入。**

<p align="center">
    <img width="25%" src="https://user-images.githubusercontent.com/30433053/63404874-68059900-c417-11e9-93a2-4b91e09b1ce4.gif" style="max-width:25%;">
    </a>
</p>

>在上图中，输入为 2x2, 经反卷积操作后得到 4x4 的形状。

```python
import tensorflow as tf

y = tf.constant(1., shape=[1,2,2,1])
w = tf.constant(1., shape=[3,3,1,1])
x = tf.nn.conv2d_transpose(y, w, output_shape=[1,4,4,1], strides=[1,1,1,1], padding="VALID")
print(x.shape) # [1, 4, 4, 1]
```

看完上面这个过程，你还会觉得反卷积操作难以理解吗？

# One more thing

现在好像对反卷积操作的理解比较清醒了点，说白了就是调用 tf.nn.conv2d_transpose 操作然后把 tf.nn.conv2d 过程里的输入和输出进行颠倒就是了呗。下面来看看一些其他常见的卷积和反卷积操作是怎样的。

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td>op参数</td>
    <td>padding="VALID", strides=2</td>
    <td>padding="SAME", strides=2</td>
    <td>padding="SAME", strides=1</td>
  </tr>
  <tr>
    <td>卷积过程</td>
    <td><img width="150px" src="https://user-images.githubusercontent.com/30433053/63411725-bd967180-c428-11e9-8858-ef1058f9c490.gif"></td>
    <td><img width="150px" src="https://user-images.githubusercontent.com/30433053/63412415-4d88eb00-c42a-11e9-8338-546efc29636d.gif"></td>
    <td><img width="150px" src="https://user-images.githubusercontent.com/30433053/63412815-0a7b4780-c42b-11e9-99d2-a9e3d95fd907.gif"></td>
  </tr>
    <td>反卷积过程</td>
    <td><img width="150px" src="https://user-images.githubusercontent.com/30433053/63412214-d5bac080-c429-11e9-8e0f-89180c14ab6e.gif"></td>
    <td><img width="150px" src="https://user-images.githubusercontent.com/30433053/63412567-90e35980-c42a-11e9-9f68-1d793536599a.gif"></td>
    <td><img width="150px" src="https://user-images.githubusercontent.com/30433053/63413031-7eb5eb00-c42b-11e9-9b37-4e28ec970365.gif"></td>
  </tr>


>看完下面这些过程，就会觉得反卷积其实就是一种特殊的卷积操作而已（**先对输入矩阵进行各种填充以扩大，然后再对此进行卷积操作**）。
