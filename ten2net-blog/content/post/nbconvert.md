+++
type = "post"
date = "2016-12-22T12:59:11+08:00"
tags = ["Jupyter","nbconvert","Hugo"]
author = "王峰"
draft = false
description = "本篇介绍了使用jupyter nbconvert转换jupyter notebook笔记为其它格式的方法。如果要发布博客，你可以使用该工具转换为Markdown格式后利用Hugo发布。"
title = "使用jupyter nbconvert转换jupyter notebook笔记为Markdown格式"
topics = ["格式转换"]
keywords = ["Jupyter","nbconvert"]

+++

# nbconvert
### Jupyter Notebook Conversion

[![Google Group](https://img.shields.io/badge/-Google%20Group-lightgrey.svg)](https://groups.google.com/forum/#!forum/jupyter)
[![Build Status](https://travis-ci.org/jupyter/nbconvert.svg?branch=master)](https://travis-ci.org/jupyter/nbconvert)
[![Documentation Status](https://readthedocs.org/projects/nbconvert/badge/?version=latest)](https://nbconvert.readthedocs.io/en/latest/?badge=latest)
[![Documentation Status](https://readthedocs.org/projects/nbconvert/badge/?version=stable)](http://nbconvert.readthedocs.io/en/stable/?badge=stable)
[![codecov.io](https://codecov.io/github/jupyter/nbconvert/coverage.svg?branch=master)](https://codecov.io/github/jupyter/nbconvert?branch=master)

## 用法
```
    $ jupyter nbconvert --to <output format> <input notebook>
```
- 其中：<output format>`可以是下面几种：

-* HTML
-* LaTeX
-* PDF
-* Reveal JS
-* Markdown (md)
-* ReStructured Text (rst)
-* executable script

### 例子: Convert a notebook to HTML
```
    $ jupyter nbconvert --to html mynotebook.ipynb
```
This command creates an HTML output file named `mynotebook.html`.

## 资源

- [Documentation for Jupyter nbconvert](https://nbconvert.readthedocs.io/en/latest/)
  [[PDF](https://media.readthedocs.org/pdf/nbconvert/latest/nbconvert.pdf)]
- [nbconvert examples on GitHub](https://github.com/jupyter/nbconvert-examples)
- [Issues](https://github.com/jupyter/nbconvert/issues)
- [Technical support - Jupyter Google Group](https://groups.google.com/forum/#!forum/jupyter)
- [Project Jupyter website](https://jupyter.org)
- [Documentation for Project Jupyter](https://jupyter.readthedocs.io/en/latest/index.html)
  [[PDF](https://media.readthedocs.org/pdf/jupyter/latest/jupyter.pdf)]


[Jinja]: http://jinja.pocoo.org/