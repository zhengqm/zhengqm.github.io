---
layout: post
title: Surprising JavaScript I - Scope & Closures
categories:
- blog
tags:
- code
---

写过JavaScript代码，用过前端框架，但是对JavaScript自身的语言特性一直没有很好的理解。在很长一段时间中我只了解JavaScript的基本语法，并在以一种 stackoverflow-oriented 的方式写代码。今天读完了[《You Don't Know JS: Scope & Closures》](https://github.com/getify/You-Dont-Know-JS/tree/master/scope%20%26%20closures)，了解到了许多出乎我意料的语言细节，记录于此。

# Scope

对变量赋值时，会从当前到scope到全局scope逐步寻找变量的声明，如果一直找不到的话将变量添加到全局scope中。从内到外寻找变量声明非常容易接受，但是将变量直接往全局scope里面扔意味着一些意想不到的副作用：

```javascript
console.log(item) // ReferenceError
(function(){item = 1})()
console.log(item) // 1 -- 全局变量  :(
```
如果上面这片代码太简单粗暴的话，不妨看看这个：

```javascript
function foo(obj) {
    with (obj) {
        a = 2;
    }
}

var o1 = {
    a: 3
};

var o2 = {
    b: 3
};

foo( o1 );
console.log( o1.a ); // 2

foo( o2 );
console.log( o2.a ); // undefined
console.log( a ); // 2 -- 全局变量  :(
```

在调用 `foo(o2)` 时，解释器首先在 `o2` 里找变量 `a` ，发现找不到于是在上一层scope即全局scope中找变量 `a` ，发现还是找不到，于是直接把变量 `a` 加到全局scope中，并赋值为 `2`...

如果你惊到了，说明你也遇到了“意想不到的副作用”。

# Function Scope & Block Scope

猜猜看，下面这片代码中`bar`、`bam`、`baz`、`i` 的scope是什么，在函数哪里可见。

```javascript
function foo() {
    var bar = 2;
    if (bar > 1 || bam) {
        var baz = bar * 10;
    }
    
    for(var i = 0; i < 10; i++ ){
        baz += 1;
    }

    var bam = (baz * 2) + 2;

    console.log( bam );
}
```

如果你说 `bam` 在被声明后可见，`i` 的scope只限于循环内部，你就猜错了。所有这些变量在整个函数中都是可见的，它们的scope都是 `foo` 函数内部。原因是JavaScript并不支持我们通常意义下的block scope，JavaScript只支持最简单的function scope，即所有scope均以函数为最小单位。

这当然是不合理的，所有程序设计课程都告诉我们要尽可能隐藏细节，当 `i` 只用来完成循环的时候，没理由让整个函数都看到这个变量。于是在新标准中，我们有了关键字 `let`，为JavaScript添加了block scope支持：

```javascript
function foo(n){
	console.log(bar); // 不可见
	
	if (n > 10){
		let bar = 5;
		console.log(bar) //可见
	}
	
	console.log(bar) //不可见
}
```

除了支持block scope外，`let` 还把循环变量scope到了循环的每一次迭代中。在下面这片代码中：

```javascript
for (var i=1; i<=5; i++) {
    setTimeout(function(){
        console.log("i:",i);
    },i*1000);
}

for (let i=1; i<=5; i++) {
    setTimeout(function(){
        console.log("i:",i);
    },i*1000);
}

```

第一个循环的输出是:

```
i:6
i:6
i:6
i:6
i:6
```

而第二个循环的输出是：

```
i:1
i:2
i:3
i:4
i:5
```

这是一个很棒的特性！早点知道这个特性的话，上个学期用D3.js写的大作业的代码应该会干净不少。

# Hoisting


猜猜看：下面这片代码的输出是什么？

```javascript
a = 2;
var a;
console.log( a );
```

答案是 `2` 而非 `undefined`，


继续猜：下面这片代码的输出是什么？

```javascript
console.log( a );
var a = 2;
```

答案是 `undefined` 而非 `ReferenceError`。

事情的缘由是解释器会把变量的声明“提前”(hoisting)，上述代码变成了：

```javascript
var a;
a = 2;
console.log( a );
```

以及

```javascript
var a;
console.log( a );
a = 2;
```

所以输出自然是 `2` 和 `undefined` 了。

# Closure

所谓Closure,就是当一个函数在它的lexical scope之外被执行时，它仍能够访问它lexical scope内的变量。这并不难理解，但是当Closure和循环组合在一起时就要小心了，以之前使用过的一片代码为例：

```javascript
for (var i=1; i<=5; i++) {
    setTimeout(function(){
        console.log("i:",i);
    },i*1000);
}
```

我们期望它的输出是：

```
i:1
i:2
i:3
i:4
i:5
```

它的输出为：

```
i:6
i:6
i:6
i:6
i:6
```


上述输出的原因是回调函数实际捕捉的是一个共享的scope，在这个scope中，只有一个变量 `i`，而在执行时该变量的值已被置为 `6`，所以得到了我们看到的输出。
为了得到我们希望的输出，我们需要在每次迭代中创建一个新的scope：

```javascript
for (var i=1; i<=5; i++) {
    (function(j){
        setTimeout( function timer(){
            console.log( j );
        }, j*1000 );
    })( i );
}
```

这样每次迭代的回调函数都能捕捉一个不同的scope，我们也得到了希望得到的输出。

# More

除上述内容外，[《You Don't Know JS: Scope & Closures》](https://github.com/getify/You-Dont-Know-JS/tree/master/scope%20%26%20closures)还介绍了许多JavaScript中与Scope及Closure相关的语言特性，如果你和我一样一直靠Stackoverflow写JavaScript代码，想了解更多的语言细节又不想啃《JavaScript权威指南》这种大部头，这本书绝对值得一读。


