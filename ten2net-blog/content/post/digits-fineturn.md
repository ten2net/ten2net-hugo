+++
date = "2016-12-21T17:45:00+08:00"
tags = ["digits","caffe"]
author = "author"
draft = false
type = "post"
title = "使用digits进行finetune"
topics = ["topic 1"]
keywords = ["key","words"]
description = "description"

+++

# 一、下载model参数
> 可以直接在浏览器里输入地址下载，也可以运行脚本文件下载。下载地址为：http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
文件名称为：bvlc_reference_caffenet.caffemodel，文件大小为230M左右，为了代码的统一，将这个caffemodel文件下载到caffe根目录下的 models/bvlc_reference_caffenet/ 文件夹下面。也可以运行脚本文件进行下载：
```sh
# sudo ./scripts/download_model_binary.py models/bvlc_reference_caffenet
```
# 二、准备数据
> 将训练数据放在一个文件夹内。比如我在当前用户根目录下创建了一个data文件夹，专门用来存放数据，因此我的训练图片路径为：/home/xxx/data/re/train
打开浏览器，运行digits，新建一个classification dataset,设置如下图：

