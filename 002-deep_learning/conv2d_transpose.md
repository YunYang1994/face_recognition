看了很多反卷积的文章，只想说写的都是一坨翔。关于反卷积操作，大家不要想得那么复杂，说白了其实就是**卷积操作的一个逆过程！**

★ 卷积操作


```python
import tensorflow as tf

x = tf.constant(1., shape=[1,4,4,1])
w = tf.constant(1., shape=[3,3,1,1])
y = tf.nn.conv2d(x, w, strides=[1,1,1,1,], padding='VALID')
print(y.shape) # [1, 2, 2, 1]
```


<p align="center">
    <img width="25%" src="https://user-images.githubusercontent.com/30433053/63404840-4dcbbb00-c417-11e9-8d35-0eea90c5a3c6.gif" style="max-width:25%;">
    </a>
</p>

★ 反卷积操作

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
