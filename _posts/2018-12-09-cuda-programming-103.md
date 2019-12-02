---
layout: post
title: Cuda编程103
subtitle: Cuda多卡编程简介
categories:
- blog
tags:
- code
---

系列文章：
+ [Cuda编程101](/blog/2018/11/25/cuda-programming-101/)：Cuda编程的基本概念及编程模型
+ [Cuda编程102](/blog/2018/12/02/cuda-programming-102/)：Cuda程序性能相关话题
+ [Cuda编程103](/blog/2018/12/09/cuda-programming-103/)：Cuda多卡编程
+ [Cuda tips](/blog/2018/12/07/cuda-nvcc-tips/): nvcc的`-code`、`-arch`、`-gencode`选项

# 为何使用多卡

对于一个计算问题，我们编写多卡程序对其进行的解决的好处主要包括：
+ 目前单张显卡的显存仍然偏小，使用多卡将有可能解决单卡无法解决的问题。
+ 对于单卡能够解决的问题，多卡能够提供更多的计算力及内存带宽，使用多卡有可能对问题解决提供加速。

# 问题分类

在编写多卡程序时，我们尝试解决的问题可大致分为两类：
+ 在完成问题切分为自问题的过程后，在各张卡上解决子问题的过程中，各显卡无需再进行交互或数据传输，比如常见的各种[embarrasingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel)问题。
+ 在各张卡上解决子问题的过程中，各显卡需要额外的数据交互过程。

对于第一类问题，我们只需要了解如何使用CUDA提供的api完成不同显卡的内存分配、数据传输、kernel启动；而对于第二类问题，我们还需要完成显卡间的通信问题。

# 多卡管理：内存分配、数据传输、kernel执行

CUDA程序对多卡的管理主要依赖于下列两个函数：
+ `cudaError_t cudaGetDeviceCount(int* count)`：用于查询当前可使用的显卡数量
+ `cudaError_t cudaSetDevice(int id)`：用于选择当前所需操作的显卡
 
CUDA API为程序员暴露了一个状态机，程序员使用`cudaSetDevice`设置所需操作的显卡，后续的所有操作均作用于该显卡。当需要操作另一张显卡时，再使用`cudaSetDevice`进行设定。

当我们使用`cudaSetDevice`设定显卡后，所有CUDA相关操作均将作用于该显卡：
+ `cudaMalloc`将在所选择显卡上进行分配
+ 所创建的`stream`将与所选择显卡进行绑定
+ 所启动的kernel将在所选择显卡上执行

一个简易的流程是：

```c++
for (size_t i = 0; i < device_num; ++i) {
    cudaDevice(i);
    cudaMemcpyAsync(d_ptr[i], h_ptr[i], size, cudaMemcpyHostToDevice);
    kernel<<<grid, block>>>(...);
    cudaMemcpyAsync(h_ptr[i], d_ptr[i], size, cudaMemcpyDeviceToHost);
}
```
在上述示意程序中，在每个循环内完成了显卡设定、数据传输、kernel启动的步骤，由于我们`cudaDevice`调用是非阻塞的、我们选择了`cudaMemcpyAsync`完成数据传输、kernel的启动也是非阻塞的，上述循环将迅速完成不会被cuda调用阻塞。

# 多卡间通信

当多卡间需要进行通信时，最显然的办法是将数据从来源卡拷贝至cpu内存中，再从cpu内存拷贝至目标卡中。当两张卡连接至同一PCIe总线时，两张gpu卡可直接进行p2p通信而无需通过cpu内存中转，所需要的CUDA调用主要是：

```c++
cudaError_t
cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
```
用于检查`device`设备是否能够直接访问`peerDevice`设备。

```c++
cudaError_t 
cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flag);
```
用于允许当先所选定显卡访问`peerDevice`上的数据

```c++
cudaError_t
cudaMemcpyPeerAsync(void* dst, int dstDev,
                    void* src, int srcDev, 
                    size_t nBytes, cudaStream_t stream);
```
用于从`srcDev`显卡的`src`内存地址拷贝`nBytes`数据至`dstDev`显卡的`dst`位置。

基于这些CUDA调用，程序员能够在多卡上启动不同的kernel，并通过各卡上kernel所在的stream判断kernel的执行状态，并在执行结束后通过p2p调用完成数据的传输并进行下一阶段的计算。