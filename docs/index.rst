.. MNN-Doc documentation master file, created by
   sphinx-quickstart on Wed Aug 17 10:39:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎使用MNN文档
===================================

遇到问题请先查看文档和FAQ，如果没有答案请在Github提issue或在钉钉群提问。

.. toctree::
   :maxdepth: 1
   :caption: 介绍
   :name: introduction

   intro/about
   intro/releases

.. toctree::
   :maxdepth: 1
   :caption: 快速开始
   :name: quick-start

   start/overall
   start/demo

.. toctree::
   :maxdepth: 1
   :caption: 从源码构建
   :name: compile

   compile/cmake
   compile/engine
   compile/other
   compile/pymnn

.. toctree::
   :maxdepth: 1
   :caption: 推理用法
   :name: inference

   inference/session
   inference/module
   inference/python

.. toctree::
   :maxdepth: 1
   :caption: 表达式
   :name: expr

   inference/expr

.. toctree::
   :maxdepth: 1
   :caption: 训练框架
   :name: train

   train/expr
   train/data
   train/optim
   train/quant
   train/finetune
   train/distl

.. toctree::
   :maxdepth: 1
   :caption: 生成式模型
   :name: transformers

   transformers/diffusion
   transformers/llm

.. toctree::
   :maxdepth: 1
   :caption: 测试工具
   :name: tools

   tools/convert
   tools/test
   tools/benchmark
   tools/quant
   tools/compress
   tools/visual
   tools/python

.. toctree::
   :maxdepth: 1
   :caption: 贡献代码
   :name: contribute

   contribute/code
   contribute/backend
   contribute/op

.. toctree::
   :maxdepth: 1
   :caption: FAQ
   :name: faq

   faq.md

.. toctree::
   :maxdepth: 1
   :caption: C++ API

   cpp/Interpreter
   cpp/Tensor
   cpp/ImageProcess
   cpp/Matrix

   cpp/Expr
   cpp/Module
   cpp/Optimizer
   cpp/MathOp
   cpp/NeuralNetWorkOp

.. toctree::
   :maxdepth: 1
   :caption: Python API

   pymnn/MNN
   pymnn/expr
   pymnn/numpy
   pymnn/cv
   pymnn/nn
   pymnn/optim
   pymnn/data
   pymnn/loss
   pymnn/compress
   pymnn/linalg
   pymnn/random

   pymnn/Interpreter
   pymnn/Session
   pymnn/OpInfo
   pymnn/Tensor
   pymnn/CVImageProcess
   pymnn/CVMatrix
   pymnn/Var
   pymnn/_Module
   pymnn/RuntimeManager
   pymnn/Optimizer
   pymnn/Dataset
   pymnn/DataLoader

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
