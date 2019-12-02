---
layout: post
title: 基于Meteor实现一个的即时搜索工具
categories:
- blog
tags:
- code
---


# Introduction

## Background

我所在的小组的研究领域是软件开发数据挖掘，在各种的软件开发数据中，最基础且最重要的数据是软件项目的代码库，其中包含了项目的所有代码文件、代码的所有版本、代码提交者的想关信息。我们小组从互联网的各大开源社区、代码托管网站爬取了超过20万个软件项目的代码库，其中既包含 Apache 、 Mozilla 这种大型社区中的项目，也包含托管于 Github 、 Bitbucket 等网站上的大大小小的项目。在此基础上，小组的师兄们对这些项目的元信息进行了抽取，并将其存储于 Mongodb 中，对于一个项目我们存储了项目的:

+ 名称 (prj)
+ 所托管的网站 (repo)
+ 代码库所在位置 (src_loc)
+ 提交日志所在位置 (log_loc)
+ 提交人数 (n_peo)
+ 提交的版本数 (n_cmt)
+ 所使用的版本控制系统 (vcs)
+ 项目的起止时间 (b_time/e_time)
+ 时间跨度等信息 (span)

以 Hadoop 为例， 它在 Mongodb中的一条记录为：

```json
{ 
    "_id" : "1430277742.791925", 
    "prj" : "hadoop-common.git",   
    "repo" : "git.apache.org",    
    "src_loc" : "/path/to/datastore/git/git.apache.org_hadoop-common.git", 
    "log_loc" : "/path/to/datastore/git/git.apache.org/hadoop-common.git", 
    "n_peo" : 36, 
    "n_cmt" : 5825，
    "vcs" : "git"
    "b_time" : 200601, 
    "e_time" : 201005, 
    "span" : 52,
    "script" : "", 
}
```

然而尽管我们有了大量数据，也提取了这些数据的元信息，但是这些数据一直静静的躺在服务器的磁盘上，并没有被很好利用。其中一个重要的原因是我们缺少一个方便好用的搜索工具对数据进行探索。

## Instant Search

而所谓即时搜索(Instant search)，就是在使用者键入关键字的同时即时返回搜索结果，最为常见的例子自然是 [Google](https://www.google.com/) 和[百度](https://www.baidu.com/)了。


## Goal

今天我将要完成的目标是基于实验室存储于 Mongodb 的代码项目元数据，实现一个能够帮小组成员搜索数据的工具，它能够支持：

+ 根据用户输入进行即时搜索
+ 支持关键字的正则匹配
+ 支持条件搜索

## Framework

<div style="text-align:center;margin-bottom:30px;"><img src ="/static/meteor-logo.png" /></div>

引自[官网](https://www.meteor.com/):
>Meteor is a complete open source platform
for building web and mobile apps
in pure JavaScript.

引自[36Kr](http://36kr.com/p/99503.html):
> Meteor 是一个新鲜出炉的现代网站开发平台，基础构架是 Node.JS + MongoDB，它把这个基础构架同时延伸到了浏览器端，如果 App 用纯 JavaScript 写成，JS APIs 和 DB APIs 就可以同时在服务器端和客户端无差异地调用，本地和远程数据通过 DDP（Distributed Data Protocol）协议传输。

接下来我们开始基于 Meteor 实现上文介绍的即时搜索工具。

# In Action



## Back End

方便起见，我们将把客户端、服务器端的代码放在同一个 javascript 文件中。我们首先获取 Mongodb 中存储元数据的 collection:

```javascript
Items = new Mongo.Collection("log_info");
```

值得注意的是，这个对象既可在客户端代码中使用，也可在服务器端代码使用， Meteor 会基于 DDP 协议帮我们搞定数据在服务器、客户端之间的传输问题。

在服务器端，我们根据用户提交的关键字对数据库进行查询，并发布 (publish) 与之相关的数据：

```javascript
if (Meteor.isServer) {
  Meteor.publish('items', function (queryString) {
      return query(Items, queryString) // To implement later
  })
}
```

在客户端，当用户键入查询时，我们捕捉键盘输入事件，并将查询字符串存储于 Session 中，并根据查询字符串向服务器端订阅 (subscribe) 新数据：

```javascript
  Template.body.events({
    "keyup #search-box":_.throttle(function(event){
      Session.set('queryString', event.target.value)
      Meteor.subscribe('items', event.target.value)
    }, 50)
  })
```

另外我们需要一个帮助函数对所得数据进行展示，这个函数将会在模板中被使用:

```javascript
  Template.body.helpers({
    items: function () {
      return query(Items, Session.get('queryString'))
    }
  });
```

最后我们来实现上面尚未实现的 `query` 函数，函数所完成的工作是:

+ 根据用户的输入构造查询数据库的条件
+ 对数据库进行查询
+ 返回查询结果


```javascript
function query(collections, queryString){
  var limit = 40
  var query = queryString.split(' ')
  var andArray = []
  for (var i = query.length - 1; i >= 0; i--) {
    if (query[i] == '') {
      continue
    }
    var testSpecial  = parseSpecial(query[i])
    if (testSpecial != null) {
      andArray.push(testSpecial)
    } else {
      var regEx = new RegExp(query[i], 'ig')
      andArray.push({prj: regEx})
    }
  }

  if (andArray.length != 0){
    return collections.find({$and: andArray}, {limit:limit})
  } else {
    return collections.find({},{limit:limit})
  }
}
```

而其中的 `parseSpecial` 函数实现了我们想提供的条件搜索功能，若包含特殊操作符，则构造搜索特殊搜索条件，否则使用对项目名称进行正则匹配：

```javascript
// should be DRYer

function parseSpecial(str){
  var relation
  var result = {}
  if (str.indexOf('<') != -1){
    relation = str.split('<')
    result[relation[0]] = { $lt: Number(relation[1])}
    return result
  } else if (str.indexOf('>') != -1){
    relation = str.split('>')
    result[relation[0]] = { $gt: Number(relation[1])}
    return result
  } else if (str.indexOf('=') != -1){
    relation = str.split('=')
    result[relation[0]] = relation[1]
    return result
  } else if (str.indexOf(':') != -1){
    relation = str.split(':')
    result[relation[0]] = new RegExp(relation[1], 'ig')
    return result
  } 
  return null;
}
```
至此，后端的所有功能已经全部完成。


## Front End

在前端中，我们需要一个模板对数据进行展示：

```html
<template name="item">
    <dl class="dl-horizontal">
    <dt>Project</dt>            <dd>{% raw %}{{prj}}{% endraw %}</dd>
    <dt>Repository</dt>         <dd>{% raw %}{{repo}}{% endraw %}</dd>
    <dt>Time Span</dt>          <dd>{% raw %}{{b_time}} - {{e_time}}{% endraw %}</dd>
    <dt>Version Control</dt>    <dd>{% raw %}{{vcs}}{% endraw %}</dd>
    <dt># Commit</dt>           <dd>{% raw %}{{n_cmt}}{% endraw %}</dd>
    <dt># People</dt>           <dd>{% raw %}{{n_peo}}{% endraw %}</dd>
    <dt>Source Location</dt>    <dd>{% raw %}{{src_loc}}{% endraw %}</dd>
    <dt>Log Location</dt>       <dd>{% raw %}{{log_loc}}{% endraw %}</dd>
    <hr>
    </dl>
</template>
```

并且在 `body` 中使用这个模板：


```html
<body>
  <div class="container">
    <h1>Source Repo Search</h1>
      <form class="form-horizontal" onsubmit="return false;">
      <div class="form-group">
        <input type="text" class="form-control" id="search-box" 
        placeholder="Type project keyword to search">
      </div>
      </form>
      <div id='help'>
        <small>Possible options: repo | b_time | e_time | span | vcs | n_cmt | n_peo </small><br>
        <small>Possible operators:  &gt; | = | &lt; | : </small><br>
        <small>RegEx supported.</small>
      </div>
      {% raw %}
      {{#each items}}
        {{> item}}
      {{/each}}
      {% endraw %}
    </div> 
</body>
```

可以看到 Meteor 所使用的模板语言是非常简单直接的。另外在 `head` 中我们需要 `bootstrap` 中的样式:

```html
<head>
  <title>SRSearch</title>
  <link rel="stylesheet" href="/css/bootstrap.min.css">
  <link rel="stylesheet" href="/css/bootstrap-theme.min.css">
</head>
```

这样所有功能就完成了，所有代码（包括空行）不到120行，下面是最终效果：

![](/static/demo.gif)

项目完整代码可见于 [Github](https://github.com/zhengqm/SRSearch)。

# Summary

今天我基于Meteor 框架，只用了120行代码实现了一个即时搜索工具的前端、后端，并支持正则、条件搜索功能。可以看到用 Meteor 开发实时应用是非常高效的。考虑到这个工具只使用了 Meteor 所提供特性中的冰山一角，Meteor 所能完成的事情着实令人期待。
