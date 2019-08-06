相信很多开发者都有这样的悲痛经历，很多最新的论文研究是用 Pytorch 实现的，但是公司却要求我们用 Tensorflow 去部署。微软在2017年早就开源了 ONNX，但是一直不官方地支持与 Tensorflow 模型的转化。**于是，我利用 MNIST 数据集制作了一个简单的例子以剖析这两个框架之间是如何实现转化的。**

# 训练模型
首先，我用 pytorch 建离了一个很简单的卷积网络模型，它只有一层卷积层和全连接层。为了避免过拟合，我还加了 BatchNorm 层。

```python
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.fc = nn.Linear(28*28*16, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

torch_model = ConvNet()
```
接着我们就可以直接用 torchvision 自带的 MNIST 数据集进行训练了。

```bashrc
$ python torch_train.py
```



