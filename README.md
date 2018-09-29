
|Author|YunYang1994|
|---|---
|E-mail|dreameryangyun@sjtu.edu.cn

[<img src="image/cuda-c-programming.png" alt="logo" height="300" align="right" />](https://book.douban.com/subject/27108836/)

# 《CUDA C 编程权威指南》

![](https://img.shields.io/badge/version-v2-green.svg)
[![](https://img.shields.io/badge/language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red.svg)](./README.md)
[![](https://img.shields.io/badge/chat-%E4%BA%A4%E6%B5%81-667ed5.svg)](./assets/community.md) 

> 正在学习和使用，敬请期待。

## 内容简介

CUDA (Compute Unified Device Architecture, 统一计算设备架构) 是NIVIDIA提出的并行计算架构， 结合了CPU和GPU的优点，主要用来处理密集型及并行计算。CPU和GPU是两个独立的处理器，通过单个计算节点的 PCI-Express总线相连。**GPU 用来提高计算密集型应用程序中并行程序段的执行速度， CPU则负责管理设备端的资源。** CUDA编程的独特优势在于开放的架构特性可以使得程序员在功能强大的硬件平台上充分挖掘其并行，既满足了计算密集型的程序的需要，又实现了程序的易读性及便捷性。

GPU并不是一个独立运行的计算平台，而需要与CPU协同工作，可以看成是CPU的协处理器，因此当我们在说GPU并行计算时，其实是指的基于CPU+GPU的异构计算架构。在异构计算架构中，GPU与CPU通过PCIe总线连接在一起来协同工作，CPU所在位置称为为主机端（host），而GPU所在位置称为设备端（device），如下图所示。



## 目录结构
- [**第 1 章 基于CUDA的异构并行计算**](./src/chapter01/README.md)
- [**第 2 章 CUDA 编程模型**](./src/chapter02/README.md)
