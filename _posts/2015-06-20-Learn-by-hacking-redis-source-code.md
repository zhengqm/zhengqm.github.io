---
layout: post
title: Learn by Hacking - Redis源码速览
categories:
- blog
tags:
- code
---



# Introduction

人们常说写代码进步最快的方式之一是阅读成熟开源项目的代码，从中可以学习到许多良好的代码风格、问题抽象实践。今天我选择了下面这个代码质量被广泛认可的开源项目源码进行阅读：

![](/static/redis-logo.jpg)

[Redis](redis.io)是一个用c语言实现的key-value store。除了最基础的基于字符串的键值对，redis还支持哈希、列表、集合、有序集合等数据结构，所以redis也常被称为是一个data structure server。

我使用的redis源码版本是[redis 3.0.2](http://download.redis.io/releases/redis-3.0.2.tar.gz)。Redis的编译、运行出乎意料的简单。由于它将所有依赖项均以源码方式加入项目中，在代码根目录下一句简单的`make`就可以完成所有编译任务，再来一句`make test`就可以完成所有测试。从代码下载到完成测试，整个过程耗时竟然没有超过5分钟。

为了避免在茫茫代码中走神，我决定以实现一条简单命令的方式逐步阅读redis的代码。我打算实现的命令是`randget`，它接受一个列表名作为参数，并随机返回列表中的一个元素。这的确是一个实际用处不大的命令，但应该能帮助我们更有目的性的阅读代码。

# In Action

接下来我们正式开始实现这个随机返回数组元素的 `randget` 命令。
Redis所支持的所有命令均存储于[redis.c](https://github.com/antirez/redis/blob/unstable/src/redis.c)文件开头的 `redisCommandTable` 数组中：

```c
struct redisCommand redisCommandTable[] = {
    {"get",getCommand,2,"rF",0,NULL,1,1,1,0,0},
    {"randget",randgetCommand,2,"rF",0, NULL,1,1,1,0,0},
    {"set",setCommand,-3,"wm",0,NULL,1,1,1,0,0},
    ...
```

数组中的每个元素是一个 `redisCommand ` 结构体，结构体中记录了关于一条命令的详尽信息，可见于[redis.h](https://github.com/antirez/redis/blob/unstable/src/redis.h)，
以数组中记录的第一条命令 `get` 命令为例，它说的是：

```c
 {"get",getCommand,2,"rF",0,NULL,1,1,1,0,0}
```
	+ 命令名称为get
	+ 用于处理该命令的函数为getCommand
	+ 命令参数个数为2
	+ 是一条只读且复杂度为O(1)的命令
	+ 变量名在参数列表中的下标为1,如 `GET foo`


基于这个，我首先在数组中加入了这个条目：

```c
{"randget",randgetCommand,2,"rF",0, NULL,1,1,1,0,0},
```

由于这是个和列表相关的命令，我决定把函数 `randgetCommand` 和其他列表相关的函数放在一起。首先在 `redis.h` 加入一行声明：

```c
void randgetCommand(redisClient *c);
```

在 `t_list.c` 中加入函数定义：

```c
void randgetCommand(redisClient *c){

}
```

不妨编译一下：

```
> make
```

能够编译通过，那我们来试一试命令，先启动服务器端:

```
> src/redis-server
```

再在另外一个shell启动客户端

```
> src/redis-cli
```

并敲入：

```
127.0.0.1:6379> LPUSH mylist foo
(integer) 1
127.0.0.1:6379> randget
(error) ERR wrong number of arguments for 'randget' command
127.0.0.1:6379> randget mylist
[hang up]
```

可以看到redis能正确识别 `randget` 应接收的参数个数。当参数个数正确时，redis识别了我们敲入的命令，但由于我们还没有填入命令的实现，程序没有正确进行回应而且程序挂起了。

接下来就是在函数体中填入我们的实现了，命令需要接收一个列表名作为输入，并随机返回列表中的一个元素，抽象来说就是:

```c
void randgetCommand(redisClient *c){
	// 读入参数 
	// 使用该参数是否可取出某个列表？
	//		若否，返回空
	// 读入列表，得到列表长度 length
	// 若length为0
	// 		返回空
	// 否则
	//		随机取出[0,length)中某个整数作为下标，取出对应元素
}
```

首先是获取命令参数并进行检查，由于各个关于列表的命令都需要使用类似的操作，我们参考了其他命令中的实现：

```c
robj *o = lookupKeyReadOrReply(c,c->argv[1],shared.nullbulk);
if (o == NULL || checkType(c,o,REDIS_LIST)) return;
```

代码首先使用 `c->argv[1]` 获取列表名，并调用 `lookupKeyReadOrReply` 获取这个键对应的值，若给定的键不存在则向客户端返回空，且函数返回null。
而在第二行将检查 `o` 是否为 `null` 及 `o` 所存储的值类型是否为列表，当 `o` 为空或类型不为列表时，将向客户端返回空，并从这一命令中返回。
当
检查顺利通过时， `o` 保存的就是我们感兴趣的列表结构，我们首先获取列表的长度，若长度为0则返回空，否则随机得到一个下标并返回对应元素：

```c
    if (length == 0){
        addReply(c,shared.nullbulk);
    } else {
        long index = random() % length;
        robj *value;
    	  ...	
    }
```

Redis中列表的存储方式有两种，分别是 `Linked List` 和 `Zip List`， `Linked List`即为我们熟悉的链表，而关于 `Zip List` 是一种为了节省内存消耗而特别设计的列表结构，它的详细介绍可见于[这篇文章](http://redisbook.readthedocs.org/en/latest/compress-datastruct/ziplist.html)。我们根据列表的不同存储方式使用相应接口获取下标为 `index` 的元素：

```c
if (o->encoding == REDIS_ENCODING_ZIPLIST) {
    unsigned char *p;
    unsigned char *vstr;
    unsigned int vlen;
    long long vlong;
    p = ziplistIndex(o->ptr,index);
    if (ziplistGet(p,&vstr,&vlen,&vlong)) {
        if (vstr) {
            value = createStringObject((char*)vstr,vlen);
        } else {
            value = createStringObjectFromLongLong(vlong);
        }
        addReplyBulk(c,value);
        decrRefCount(value);
    } else {
        addReply(c,shared.nullbulk);
    }
} else if (o->encoding == REDIS_ENCODING_LINKEDLIST) {
    listNode *ln = listIndex(o->ptr,index);
    if (ln != NULL) {
        value = listNodeValue(ln);
        addReplyBulk(c,value);
    } else {
        addReply(c,shared.nullbulk);
    }
} else {
    redisPanic("Unknown list encoding");
}
```

这段代码看起来有点复杂，所完成的工作就是以获取列表中的元素，将所得元素的指针存储于 `value` 中，并通过 `addReply` 或 `addReplyBulk` 将所得元素返回。至此命令所要完成的工作就完成了，函数的全貌是：

```c
void randgetCommand(redisClient *c){
    robj *o = lookupKeyReadOrReply(c,c->argv[1],shared.nullbulk);
    if (o == NULL || checkType(c,o,REDIS_LIST)) return;
    long long length = listTypeLength(o);

    if (length == 0){
        addReply(c,shared.nullbulk);
    } else {
        long index = random() % length;
        robj *value;
        if (o->encoding == REDIS_ENCODING_ZIPLIST) {
            unsigned char *p;
            unsigned char *vstr;
            unsigned int vlen;
            long long vlong;
            p = ziplistIndex(o->ptr,index);
            if (ziplistGet(p,&vstr,&vlen,&vlong)) {
                if (vstr) {
                    value = createStringObject((char*)vstr,vlen);
                } else {
                    value = createStringObjectFromLongLong(vlong);
                }
                addReplyBulk(c,value);
                decrRefCount(value);
            } else {
                addReply(c,shared.nullbulk);
            }
        } else if (o->encoding == REDIS_ENCODING_LINKEDLIST) {
            listNode *ln = listIndex(o->ptr,index);
            if (ln != NULL) {
                value = listNodeValue(ln);
                addReplyBulk(c,value);
            } else {
                addReply(c,shared.nullbulk);
            }
        } else {
            redisPanic("Unknown list encoding");
        }
    }
    return;
}
```

接下来我们重新编译redis，并测试一下：

```
[client]
> src/redis-cli
# 初始化列表
127.0.0.1:6379> LPUSH list a
(integer) 1
127.0.0.1:6379> LPUSH list b
(integer) 2
127.0.0.1:6379> LPUSH list c
(integer) 3
127.0.0.1:6379> LPUSH list d
(integer) 4
127.0.0.1:6379> LPUSH list e
(integer) 5

# 插入完毕，开始随机获取
127.0.0.1:6379> randget list
"e"
127.0.0.1:6379> randget list
"a"
127.0.0.1:6379> randget list
"b"
127.0.0.1:6379> randget list
"d"
127.0.0.1:6379> randget list
"e"
127.0.0.1:6379> randget list
"d"

＃ 错误处理
127.0.0.1:6379> set foo bar
OK
127.0.0.1:6379> randget
(error) ERR wrong number of arguments for 'randget' command
127.0.0.1:6379> randget foo
(error) WRONGTYPE Operation against a key holding the wrong kind of value
127.0.0.1:6379> randget unknown
(nil)
127.0.0.1:6379>
```

可以看到命令能够正常工作，并且能够正确应对各种错误参数！

# 小结

今天我为redis添加了一条简单的命令，并从中了解到了redis的内部抽象及处理命令的流程。
不得不说redis代码的易读性及可扩展性做得非常非常好，我只需了解几个文件就能够轻松添加一条命令。
另外这种 learn by hacking的方式的确要比漫无目的的通读代码来得高效，值得继续。

