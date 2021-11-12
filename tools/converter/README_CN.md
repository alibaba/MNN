[English Version](README.md)

# MNNConvert

## 编译模型转换工具(gcc>=4.9)
首先需要安装protobuf(3.0以上)
```bash
# macOS
brew install protobuf
```
其它平台请参考[官方安装步骤](https://github.com/protocolbuffers/protobuf/tree/master/src)

```bash
cd MNN
mkdir build
cd build
cmake .. -DMNN_BUILD_CONVERTER=true
make
```

## 模型转换的使用

```bash
Usage:
  MNNConvert [OPTION...]

  -h, --help            Convert Other Model Format To MNN Model

  -v, --version         show current version
  -f, --framework arg   model type, ex: [TF,CAFFE,ONNX,TFLITE,MNN]
      --modelFile arg   tensorflow Pb or caffeModel, ex: *.pb,*caffemodel
      --prototxt arg    only used for caffe, ex: *.prototxt
      --MNNModel arg    MNN model, ex: *.mnn
      --benchmarkModel  Do NOT save big size data, such as Conv's weight,BN's
                        gamma,beta,mean and variance etc. Only used to test
                        the cost of the model
      --bizCode arg     MNN Model Flag, ex: MNN
      --debug           Enable debugging mode.
```

> 说明: 选项benchmarkModel将模型中例如卷积的weight，BN的mean，var等参数移除，减小转换后模型文件大小，在运行时随机初始化参数，以方便测试模型的性能。

### tensorflow/ONNX/tflite

```bash
./MNNConvert -f TF/ONNX/TFLITE --modelFile XXX.pb/XXX.onnx/XXX.tflite --MNNModel XXX.XX --bizCode XXX
```

三个选项是必须的！
例如:

```bash
./MNNConvert -f TF --modelFile path/to/mobilenetv1.pb --MNNModel model.mnn --bizCode MNN
```

### caffe

```bash
./MNNConvert -f CAFFE --modelFile XXX.caffemodel --prototxt XXX.prototxt --MNNModel XXX.XX --bizCode XXX
```

四个选项是必须的！
例如:

```bash
./MNNConvert -f CAFFE --modelFile path/to/mobilenetv1.caffemodel --prototxt path/to/mobilenetv1.prototxt --MNNModel model.mnn --bizCode MNN
```

### MNN

```bash
./MNNConvert -f MNN --modelFile XXX.mnn --MNNModel XXX.XX --bizCode XXX
```

### 查看版本号

```bash
./MNNConvert --version
```

## MNNDump2Json
将MNN模型bin文件 dump 成可读的类json格式文件，以方便对比原始模型参数

## Pytorch 模型转换
- 用Pytorch的 onnx.export 接口转换 Onnx 模型文件（参考：https://pytorch.org/docs/stable/onnx.html）

```
import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names, do_constant_folding=True)
```

- 将 Onnx 模型文件转成 MNN 模型

```
./MNNConvert -f ONNX --modelFile alexnet.onnx --MNNModel alexnet.mnn --bizCode MNN
```

