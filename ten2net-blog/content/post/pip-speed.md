+++
topics = ["pip"]
draft = false
date = "2016-12-22T13:55:34+08:00"
tags = ["pip","speed"]
author = "wangf"
type = "post"
title = "使用豆瓣镜像站点加速pip安装过程"
keywords = ["pip","mirror"]
description = "在pip.ini文件中添加index-url = https://pypi.douban.com/simple即可."

+++

### 1、创建一个pip.ini文本文件，windows下放到~\pip\目录下，Linux下放到~/.pip/目录下

### 2、修改文件内容如下：

```
[global] 
timeout = 6000
index-url = https://pypi.douban.com/simple
[install]
use-mirrors = true
mirrors = http://e.pypi.python.org
```
