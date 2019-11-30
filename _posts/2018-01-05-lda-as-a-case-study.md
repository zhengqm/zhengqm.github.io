---
layout: post
title: LDA作为概率模型、概率近似算法的Case Study
categories:
- blog
tags:
- algo
---

在看完LDA的资料后，留在脑海中的印象是：

> LDA可以看作是一个使用概率模型对语料库进行建模，并使用相关后验概率近似方法进行推断的一个case study。

其中概率模型的建立主要包括：
+ 基于对数据产生过程的假设写出依赖关系及概率分布
+ 获取数据
+ 获得模型参数的后验概率
+ 对现有数据进行探索或对新数据进行预测

后验概率近似方法主要包括：
+ MCMC (Markov chain Monte Carlo)
+ Variational Inference

它们在整个流程中扮演的角色是：

<img src="/static/modeling.svg"/>


LDA作为该流程的一个case study，具体做的事情是：


+ 对一篇文章的产生过程进行假设（文章是关于主题的分布，主题是关于词汇的分布，每个单词对应于某个主题assignment），并写出联合概率分布
+ 获得语料库
+ 获得各隐变量的后验概率分布
+ 探索各文章关于主题的分布，探索每个主题内单词的分布

<img src="/static/LDA.svg"/>

Todo:
并行化的MCMC/VI?




