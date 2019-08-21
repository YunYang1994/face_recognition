看了很多关于反卷积的文章，只想说写的都是一坨翔。关于反卷积操作，大家不要想得那么复杂，说白了其实就是**卷积操作的一个逆过程!** 反正牢记这一点就够了，在这里我不想贴一大堆任何人都看不懂的公式和矩阵理论，只想贴最简洁的代码帮你了解什么是反卷积操作。

# 卷积操作
假设有一个通道数为 1 的 feature map，它的宽和高都为 4，我们用一个 3x3 的卷积核对它进行卷积操作 (strides=1, padding="VALID")，最后我们便得到了一个宽和高都为 2 的 feature map。该过程如下:

<p align="center">
    <img width="25%" src="https://user-images.githubusercontent.com/30433053/63404840-4dcbbb00-c417-11e9-8d35-0eea90c5a3c6.gif" style="max-width:25%;">
    </a>
</p>

>在上图中，输入为 4x4, 经卷积操作后得到 2x2 的形状

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

```python
import tensorflow as tf

x = tf.constant(1., shape=[1,2,2,1])
w = tf.constant(1., shape=[3,3,1,1])
y = tf.nn.conv2d_transpose(x, w, output_shape=[1,4,4,1], strides=[1,1,1,1], padding="VALID")
print(y.shape) # [1, 4, 4, 1]
```

<p align="center">
    <img width="25%" src="https://user-images.githubusercontent.com/30433053/63404874-68059900-c417-11e9-93a2-4b91e09b1ce4.gif" style="max-width:25%;">
    </a>
</p>
