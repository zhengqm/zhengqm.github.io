---
layout: post
title: Cuda Tips：nvcc的code、arch、gencode选项
subtitle: nvcc的这些flag如何使用
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

在编译CUDA代码时，我们需要向`nvcc`提供我们想为哪个显卡架构编译我们的代码。然而由于CUDA代码特殊的编译过程，`nvcc`为我们提供了`-arch`、`-code`、`-gencode`三个不同的编译选项。这三个编译选项既有自己的功能又会相互影响，网上教程关于这三个编译选项的使用也不统一，我在一段时间内都处在试一试、试到对为止的状态。

实际上这三个选项并不复杂，在了解nvcc的两阶段编译过程后这三个编译选项的关系将变得十分好理解。本文将首先介绍nvcc的编译过程，并在此基础上分别介绍这三个编译选项的含义。

## nvcc的编译过程

nvcc采用了一种两阶段编译过程，cuda代码首先被编译为一种面向虚拟架构(virtual architecture)的ptx代码，即图中的stage1；然后在第二阶段中将ptx代码面向具体的实际架构(real architecture)编译为实际可执行的二进制代码，即图中的stage2。

![](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/graphics/virtual-architectures.png)

不同的架构包含了不同的功能，架构代码号越高，包含的功能越多。在向nvcc指定架构时，我们所指定的实际架构代码必须兼容于所指定的虚拟架构代码。

在运行时，若二进制代码可直接运行在所在显卡上，则直接运行二进制代码；否则，若文件中包含虚拟架构代码，显卡驱动会尝试将虚拟架构代码在编译时动态编译为二进制代码进行执行。

## `-arch`选项

`-arch`选项用于向nvcc指定在第一阶段中使用什么虚拟架构，可用的选项包括：
+ compute_30, compute_32
+ compute_35
+ compute_50, compute_52, compute_53
+ compute_60, compute_61, compute_62
+ compute_70, compute_72
+ compute_75
+ ...

## `-code`选项

`-code`选项用于向nvcc将什么代码放入最后的目标文件中。

它可用于指定nvcc在第二阶段中使用什么实际架构，可用的选项包括：
+ sm_30, sm_32
+ sm_35
+ sm_50, sm_52, sm_53
+ sm_60, sm_61, sm_62
+ sm_70, sm_72
+ sm_75
+ ...

也可用于指定面向哪种虚拟架构优化ptx代码并放入目标文件中，可用的选项同`-arch`。

## `-arch`配合`-code`

`-arch`和`-code`选项的组合可以指定nvcc在编译过程中使用的虚拟架构，以及目标文件中包含哪些虚拟架构代码及实际架构代码，比方说：

```
-arch=compute_20 -code=sm_20
```
nvcc将以`compute_20`为虚拟架构产生ptx代码，所产生的目标文件将包含面向`sm_20`实际架构的二进制代码。

```
-arch=compute_20 -code=compute_20,sm_20,sm_21
```
nvcc将以`compute_20`为虚拟架构产生ptx代码，将包含面向`compute_20`虚拟架构的ptx代码及面向`sm_20`、`sm_21`实际架构的二进制代码。

## `-gencode`选项

在使用`-arch`和`-code`时，我们能够指定不同的实际架构，但是只能指定一种虚拟架构。有时候我们希望nvcc在编译过程中使用不同的虚拟架构，并在目标文件中包含面向多种虚拟架构的ptx代码，以及面向多种实际架构的二进制代码，此时我们可以使用`-gencode`达成这一目标，比方说

```
-gencode=arch=compute_50,code=sm_50
-gencode=arch=compute_52,code=sm_52
-gencode=arch=compute_60,code=sm_60
-gencode=arch=compute_61,code=sm_61
-gencode=arch=compute_61,code=compute_61
```

目标文件将包含：
+ 基于`compute_50`ptx代码产生的`sm_50`二进制代码
+ 基于`compute_52`ptx代码产生的`sm_52`二进制代码
+ 基于`compute_60`ptx代码产生的`sm_60`二进制代码
+ 基于`compute_61`ptx代码产生的`sm_61`二进制代码
+ `compute_61`ptx代码


## 引用
+ [NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
+ [How to specify architecture to compile CUDA code](https://codeyarns.com/2014/03/03/how-to-specify-architecture-to-compile-cuda-code/)