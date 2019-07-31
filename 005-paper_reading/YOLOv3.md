既然代码贴出来了，大家又这么喜欢问，那我就讲讲我对YOLOv3算法的一点点理解吧。万字长文，希望能帮助大家。
翻了下大家开的几百条[issues](https://github.com/YunYang1994/tensorflow-yolov3/issues)，其中的吐槽可以总结成以下三点:

- **YOLOv3算法的前向传播过程怎么进行的，如何理解画网格？** 
- **YOLOv3算法是怎么训练的，损失函数理解太难了，代码写得跟一坨屎一样!** 
- **为什么我在训练的时候loss出现了Nan，有什么办法解决它吗？** 

本文的目的，就在于此。

--------------------
### 1. YOLOv3算法的前向传播过程

#### 1.1 不妨先给图片画网格

![image](https://user-images.githubusercontent.com/30433053/62187018-97863000-b39a-11e9-84ff-d7d3166f0407.png)

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