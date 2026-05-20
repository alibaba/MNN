.. MNN-Doc documentation master file, created by
   sphinx-quickstart on Wed Aug 17 10:39:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎使用MNN文档
===================================

MNN 是一个高效、轻量的深度学习推理引擎，支持 CNN / Transformer / LLM / Diffusion 等模型，覆盖移动端、服务器等多种平台。

遇到问题请先查看文档和FAQ，如果没有答案请在Github提issue或在钉钉群提问。

.. toctree::
   :maxdepth: 1
   :caption: 介绍

   intro/about

.. toctree::
   :maxdepth: 1
   :caption: 快速开始

   start/overall
   start/quickstart_python
   start/quickstart_cpp
   start/demo

.. toctree::
   :maxdepth: 1
   :caption: LLM 部署

   transformers/llm
   transformers/tokenizer
   transformers/models

.. toctree::
   :maxdepth: 1
   :caption: Diffusion 部署

   transformers/diffusion

.. toctree::
   :maxdepth: 1
   :caption: 模型转换与优化

   tools/convert
   tools/compress
   tools/quant
   tools/benchmark
   tools/test
   tools/visual
   tools/python

.. toctree::
   :maxdepth: 1
   :caption: 推理用法

   inference/module
   inference/session
   inference/expr
   inference/python
   start/python
   inference/npu

.. toctree::
   :maxdepth: 1
   :caption: 从源码构建

   compile/engine
   compile/cmake
   compile/other
   compile/pymnn

.. toctree::
   :maxdepth: 1
   :caption: 训练（实验性）

   train/expr
   train/data
   train/optim
   train/finetune
   train/distl

.. toctree::
   :maxdepth: 1
   :caption: 贡献指南

   contribute/code
   contribute/op
   contribute/backend

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq.md

.. toctree::
   :maxdepth: 1
   :caption: C++ API

   cpp/Module
   cpp/Interpreter
   cpp/Tensor
   cpp/Expr
   cpp/ImageProcess
   cpp/Matrix
   cpp/MathOp
   cpp/NeuralNetWorkOp
   cpp/Optimizer

.. toctree::
   :maxdepth: 1
   :caption: Python API

   pymnn/MNN
   pymnn/Var
   pymnn/expr
   pymnn/numpy
   pymnn/_Module
   pymnn/cv
   pymnn/nn
   pymnn/optim
   pymnn/data
   pymnn/loss
   pymnn/compress
   pymnn/linalg
   pymnn/random
   pymnn/llm
   pymnn/audio
   pymnn/Tensor
   pymnn/CVImageProcess
   pymnn/CVMatrix
   pymnn/OpInfo
   pymnn/RuntimeManager
   pymnn/Optimizer
   pymnn/Dataset
   pymnn/DataLoader

.. toctree::
   :maxdepth: 1
   :caption: Python API（已废弃）

   pymnn/Interpreter
   pymnn/Session

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`