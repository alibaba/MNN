# 快速开始

## 使用MNN整体流程
在端侧应用MNN，大致可以分为三个阶段：
![concept.png](../_static/images/start/concept.png)
### 训练
在训练框架上，根据训练数据训练出模型的阶段。虽然当前MNN也提供了[训练模型的能力](../train/expr.md)，但主要用于端侧训练或模型调优。在数据量较大时，依然建议使用成熟的训练框架，如TensorFlow、PyTorch等。除了自行训练外，也可以直接利用开源的预训练模型。
### 转换
将其他训练框架模型转换为MNN模型的阶段。MNN当前支持Tensorflow(Lite)、Caffe、ONNX和TorchScript的模型转换。模型转换工具可以参考[使用说明](../tools/convert.md)；在遇到不支持的算子时，可以尝试[自定义算子](../contribute/op.md)，或在Github上给我们[提交issue](https://github.com/alibaba/MNN/issues/74)。除模型转换外，MNN也提供了[模型压缩工具](../tools/compress.md)，可以对浮点模型进行量化压缩。
### 推理
在端侧加载MNN模型进行推理的阶段。端侧运行库的编译请参考[编译文档](../compile/engine.md)（覆盖 iOS、Android、Linux/macOS、Windows 等平台）。推理接口推荐使用 [Module API](../inference/module.md)（高层接口，支持控制流），也可以使用 [Session API](../inference/session.md)（低层接口）。`demo/exec`下提供了使用示例，如图像识别 `demo/exec/pictureRecognition.cpp` ，图像实例分割（人像分割）`demo/exec/segment.cpp`，[更多demo](demo.md)。此外，[测试工具](../tools/test.md)和[benchmark工具](../tools/benchmark.md)也可以用于问题定位。
