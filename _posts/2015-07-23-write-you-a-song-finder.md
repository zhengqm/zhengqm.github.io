---
layout: post
title: 动手搭一个音乐搜索demo
categories:
- blog
tags:
- code
---

![](/static/music.jpg)

之前一段时间读到了[这篇博客](http://www.redcode.nl/blog/2010/06/creating-shazam-in-java/)，其中描述了作者如何用java实现国外著名音乐搜索工具[shazam](http://www.shazam.com/music/web/home.html)的基本功能。其中所提到的文章又将我引向了[关于shazam的一篇论文](http://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)及[另外一篇博客](https://laplacian.wordpress.com/2009/01/10/how-shazam-works/)。读完之后发现其中的原理并不十分复杂，但是方法对噪音的健壮性却非常好，出于好奇决定自己用python自己实现了一个简单的音乐搜索工具—— Song Finder, 它的核心功能被封装在 `SFEngine` 中，第三方依赖方面只使用到了 `scipy`。


### 工具demo

这个demo在ipython下展示工具的使用，本项目名称为[Song Finder](https://github.com/zhengqm/SongFinder),我把索引、搜索的功能全部封装在[Song Finder](https://github.com/zhengqm/SongFinder)中的`SFEngine`中。首先是简单的准备工作：

``` python
In [1]: from SFEngine import *

In [2]: engine = SFEngine()
```

在这之后我们对现有歌曲进行索引，我在`original`目录下准备了几十首歌曲(.wav文件)作为曲库：

```
In [3]: engine.index('original') # 索引该目录下的所有歌曲
```

在完成索引之后我们向`Song Finder`提交一段有背景噪音的歌曲录音进行搜索。对于这段《枫》在1分15秒左右的录音：

<audio controls="controls">
  <source src="/static/record0.wav" type="audio/wav" />
</audio>

工具的返回结果是：

``` python
In [4]: engine.search('record/record0.wav')
original/周杰伦-枫 73
original/周杰伦-枫 31
original/周杰伦-枫 10
original/周杰伦-枫 28
original/我要快樂 - 張惠妹 28
```

其中展示的分别是歌曲名称及片段在歌曲中出现的位置（以秒计），可以看到工具正确找回了歌曲的曲名，也找到了其在歌曲中的正确位置。

而对于这段《童话》在1分05秒左右的背景噪音更加嘈杂的录音：

<audio controls="controls">
  <source src="/static/record8.wav" type="audio/wav" />
</audio>

工具的返回结果是：

```python
In [5]: engine.search('record/record8.wav')
original/光良 - 童话 67
original/光良 - 童话 39
original/光良 - 童话 33
original/光良 - 童话 135
original/光良 - 童话 69
```

可以看到尽管噪音非常嘈杂，但是工具仍然能成功识别所对应的歌曲并对应到歌曲的正确位置，说明工具在噪音较大的环境下有良好的健壮性！

项目主页： [Github](https://github.com/zhengqm/SongFinder)


### Song Finder原理

给定曲库对一个录音片段进行检索是一个不折不扣的搜索问题，但是对音频的搜索并不像对文档、数据的搜索那么直接。为了完成对音乐的搜索，工具需要完成下列3个任务：

+ 对曲库中的所有歌曲抽取特征
+ 以相同的方式对录音片段提取特征
+ 根据录音片段的特征对曲库进行搜索，返回最相似的歌曲及其在歌曲中的位置

### 特征提取？离散傅立叶变换！

为了对音乐（音频）提取特征，一个很直接的想法是得到音乐的音高的信息，而音高在物理上对应的则又是波的频率信息。为了获取这类信息，一个非常直接的额做法是使用离散傅叶变化对声音进行分析，即使用一个滑动窗口对声音进行采样，对窗口内的数据进行离散傅立叶变化，将时间域上的信息变换为频率域上的信息，使用`scipy`的接口可以很轻松的完成。在这之后我们将频率分段，提取每频率中振幅最大的频率：

```python
def extract_feature(self, scaled, start, interval):
    end = start + interval
    dst = fft(scaled[start: end]) 
    length = len(dst)/2  
    normalized = abs(dst[:(length-1)])
    feature = [ normalized[:50].argmax(), \
                50 +  normalized[50:100].argmax(), \
                100 + normalized[100:200].argmax(), \
                200 + normalized[200:300].argmax(), \
                300 + normalized[300:400].argmax(), \
                400 + normalized[400:].argmax()]
    return feature
```
这样，对于一个滑动窗口，我提取到了6个频率作为其特征。对于整段音频，我们重复调用这个函数进行特征抽取：

```python
def sample(self, filename, start_second, duration = 5, callback = None):
    
    start = start_second * 44100
    if duration == 0:
        end = 1e15
    else:
        end = start + 44100 * duration
    interval = 8192
    scaled = self.read_and_scale(filename)
    length = scaled.size
    while start < min(length, end):
        feature = self.extract_feature(scaled, start, interval)
        if callback != None:
            callback(filename, start, feature)
        start += interval
```

其中44100为音频文件自身的采样频率，8192是我设定的取样窗口（对，这样hardcode是很不对的），`callback`是一个传入的函数，需要这个参数是因为在不同场景下对于所得到的特征会有不同的后续操作。


### 匹配曲库

在得到歌曲、录音的大量特征后，如何进行高效搜索是一个问题。一个有效的做法是建立一个特殊的哈希表，其中的key是频率，其对应的value是一系列`(曲名,时间)`的tuple，其记录的是某一歌曲在某一时间出现了某一特征频率，但是以频率为key而非以曲名或时间为key。

表格。。

这样做的好处是，当在录音中提取到某一个特征频率时，我们可以从这个哈希表中找出与该特征频率相关的歌曲及时间！

当然有了这个哈希表还不够用，我们不可能把所有与特征频率相关的歌曲都抽出来，看看谁命中的次数多，因为这样会完全无视歌曲的时序信息，并引入一些错误的匹配。

我们的做法是，对于录音中在`t`时间点的一个特征频率`f`，从曲库找出所有与`f`相关的`(曲名,时间)` tuple，例如我们得到了 

```
[(s1, t1), (s2, t2), (s3, t3)]
```

我们使用时间进行对齐，得到这个列表 

```
[(s1, t1-t), (s2, t2-t), (s3, t3-t)]
```

记为

```
[(s1, t1`), (s2, t2`), (s3, t3`)]
```

我们对所有时间点的所有特征频率均做上述操作，得到了一个大列表：


```
[(s1, t1`), (s2, t2`), (s3, t3`), ..., (sn, tn`)]
```

对这个列表进行计数，可以看到哪首歌曲的哪个时间点命中的次数最多，并将命中次数最多的`(曲名，时间)`对返回给用户。

### 不足

这个小工具是一个几个小时写成的hack，有许都地方需要改进，例如：

+ 目前只支持了wav格式的曲库及录音
+ 所有数据都放在内存中，曲库体积增大时需要引入更好的后端存储
+ 索引应该并行化，匹配也应该并行化，匹配的模型其实是典型的map-reduce。

### 项目主页

[Github](https://github.com/zhengqm/SongFinder)
