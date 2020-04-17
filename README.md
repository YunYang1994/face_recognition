我本人呢，非常喜欢编程和深度学习，因此把别人喝咖啡的功夫用在了这个仓库上。里面的代码包含了大量的中文注释，并且每一个操作都是底层实现，没有调用任何第三方库。

我是强烈建议你下载它的，如果你发现了任何的错误或有更好的意见，欢迎提 issue 或者给我发邮件: dreameryangyun@sjtu.edu.cn

⏳ 更新目录
--------------------

- [x] 2020-04-11 支持读写图片，构建基础图像类 Image，详见 [example01.cpp](https://github.com/YunYang1994/yynet/blob/master/examples/example01.cpp)

⚙️ 使用方法
--------------------

- 下载 mnist 数据
```bashrc
$ wget https://pjreddie.com/media/files/mnist.tar.gz
$ tar xvzf mnist.tar.gz
```
- 编译 example 程序
```bashrc
$ make example01
$ ./example01 images/sample.png 3
```
