相信很多开发者都有这样的悲痛经历，很多最新的论文研究是用 Pytorch 实现的，但是公司却要求我们用 Tensorflow 去部署。微软在2017年早就开源了 ONNX，但是一直不官方地支持与 Tensorflow 模型的转化。**于是，我利用 MNIST 数据集制作了一个简单的例子以剖析这两个框架之间是如何实现转化的。**

# torch_model.py
我建了一个很简单的网络，只有一层卷积层和全连接层。为了避免过拟合，我还加了 BatchNorm 层。

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

# torch_train.py
下面就是简单地进行训练了，直接运行命令 `python torch_train.py` 即可。

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch_model import torch_model

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 50
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# Prepare model
model = torch_model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

test_dataset = torchvision.datasets.MNIST(root='./',
                                          train=False,
                                          transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)
# Test the model
torch_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = torch_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('=> Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

torch.save(model.state_dict(), 'model.pth')
```