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

# 权重转化
先将 pytorch 模型的每一层权重按照顺序解析出来，这里需要特别注意的是 pytorch 模型的每一层输入和输出的形状都是 (batch_size, channels, height, width) , 而 tensorflow 的却是 (batch_size, height, width, channels)。因此，需要对所有的卷积层和全连接层的权重都需要进行转置。

```python
conv_weight = torch_conv_layer.weight.detach().numpy()
conv_weight = np.transpose(conv_weight, [2, 3, 1, 0]) # 转置卷积层权重
bias_weight = torch_conv_layer.bias.detach().numpy()
conv_layer_weights = [conv_weight, bias_weight]

gama = torch_bn_layer.weight.detach().numpy()
beta = torch_bn_layer.bias.detach().numpy()
running_mean = torch_bn_layer.running_mean.detach().numpy()
running_var = torch_bn_layer.running_var.detach().numpy()
bn_layer_weights = [gama, beta, running_mean, running_var]

linear_weight = torch_Linear_layer.weight.detach().numpy()
linear_weight = np.transpose(linear_weight, [1, 0])
linear_bias = torch_Linear_layer.bias.detach().numpy()
linear_layer_weights = [linear_weight, linear_bias]
```

