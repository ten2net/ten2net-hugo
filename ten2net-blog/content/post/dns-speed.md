+++
title = "使用http://tool.chinaz.com/dns查找加速github.com访问速度的方法"
tags = ["DNS","Speed"]
topics = ["topic 1"]
author = "wangf"
date = "2016-12-22T13:41:05+08:00"
keywords = ["DNS","Speed"]
description = "description"
draft = true
type = "post"

+++


# 为什么慢？github的CDN被某墙屏了。
>  有vpn服务的可以直接使用vpn，没有vpn的，可以绕过dns解析，在本地直接绑定host。方法如下：

- 打开dns查询工具网站：http://tool.chinaz.com/dns 
- 查询域名解析如下：
……

- 选取一个TTL值最小的ip，直接绑定到hosts文件便可解决，比如我选取第一个ip，绑定域名如下
```103.245.222.133 assets-cdn.github.com```
- 多刷几下，访问速度就一切正常了

- 域名映射：（主要是第一个）
```
    103.245.222.249 github.global.ssl.fastly.net
    103.245.222.133 assets-cdn.github.com
```
* 瞬间从打死也就20K提速到100K左右，最高能达到200多K，低时也有50K。（具体速度和个人网络环境有关，反正快了好几倍）

---
>以下来源于网络：

- github.com 上有两种源码获取方式，一是 git clone，一是直接下载 master.zip，后者明显速度快于前者，可以考虑；
- 1）用 proxychains 这类透明代理，间接走系统中运行的代理工具中转；
- 2）用 git 内置代理，直接走系统中运行的代理工具中转，比如，你的 SS 本地端口是 1080，那么可以如下方式走代理

```
git config --global http.proxy socks5://127.0.0.1:1080
git config --global https.proxy socks5://127.0.0.1:1080
```
也可以如下方式停走代理
```
git config --global http.proxy ""
git config --global https.proxy ""
```

