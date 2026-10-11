[中文版本](README_CN.md)

# MNNConvert

## Compile Model Convert Tools(gcc>=4.9)
Firstly you need to install protobuf (version>3.0)
```bash
# macOS
brew install protobuf
```
Look up the [official document of installation](https://github.com/protocolbuffers/protobuf/tree/master/src) for other platforms.

```bash
cd MNN
mkdir build
cd build
cmake .. -DMNN_BUILD_CONVERTER=true
make

# or execute the shell script directly
./build_tool.sh
```

## Usage Of Model Converter Command

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

> Note: Option benchmarkModel has removed some parameters from the model, such as weight of convolution、mean、var of BN，to reduce the size of converted model，and initialize params randomly in runtime，it will be helpful in performance testing.

### tensorflow/ONNX/tflite

```bash
./MNNConvert -f TF/ONNX/TFLITE --modelFile XXX.pb/XXX.onnx/XXX.tflite --MNNModel XXX.XX --bizCode XXX
```

These three options are necessary!
For example:

```bash
./MNNConvert -f TF --modelFile path/to/mobilenetv1.pb --MNNModel model.mnn --bizCode MNN
```

### caffe

```bash
./MNNConvert -f CAFFE --modelFile XXX.caffemodel --prototxt XXX.prototxt --MNNModel XXX.XX --bizCode XXX
```

These four options are necessary!
For example:

```bash
./MNNConvert -f CAFFE --modelFile path/to/mobilenetv1.caffemodel --prototxt path/to/mobilenetv1.prototxt --MNNModel model.mnn --bizCode MNN
```

### MNN

```bash
./MNNConvert -f MNN --modelFile XXX.mnn --MNNModel XXX.XX --bizCode XXX
```

### Show Version

```bash
./MNNConvert --version
```

## MNNDump2Json
Dump MNN binary model file to readable format like json, it will be helpful when compared to original model parameters.

## How to Convert Pytorch Model
- Turn pytorch model to Onnx (https://pytorch.org/docs/stable/onnx.html)

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

- Turn Onnx to MNN

```
./MNNConvert -f ONNX --modelFile alexnet.onnx --MNNModel alexnet.mnn --bizCode MNN
```

