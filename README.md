In this repository you will learn to build your own neural-network from scratch. To get started make sure you have `git`, a C/C++ compiler, and `make` installed. Then run:


⏳ Contents
--------------------

- [x] support to load image，more details see [example01.cpp](https://github.com/YunYang1994/yynet/blob/master/examples/example01.cpp)
- [x] support to resize, copy and gray in image processing，more details see [example02.cpp](https://github.com/YunYang1994/yynet/blob/master/examples/example02.cpp)
- [x] implement a fully connected layer to classify mnist data

⚙️ Useage
--------------------

- download mnist data
```bashrc
$ wget https://pjreddie.com/media/files/mnist.tar.gz
$ tar xvzf mnist.tar.gz
```
- make your example
```bashrc
$ make example01
$ ./example01 images/sample.png 3
```
