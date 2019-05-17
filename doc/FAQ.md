## Compiling FAQ
### Environment Requirement

cmake 3.10+
gcc 4.9+
protobuf 3.0+

__Remember to run cmake again after upgrading gcc.__


### schema/generate.sh Relative Errors

``` shell
*** building flatc ***
CMake Error: Could not find CMAKE_ROOT !!!
```

If the script fails with error above, your CMake was not installed correctly. 

Try```sudo apt install extra-cmake-modules```or```export CMAKE_ROOT=/path/to/where_cmake_installed```to fix it.

__Remember to run schema/generate.sh after editing schema (*.proto).__


### tools/script/get_model.sh Relative Errors

``` shell
Could NOT find Protobuf (missing: Protobuf_INCLUDE_DIR)
```

``` shell
Unrecognized syntax identifier "proto3".  This parser only recognizes "proto2".
```

If the script fails with errors above, your protobuf was not installed correctly. Follow [Protobuf's Installation Instructions](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) to install it.

If there are multiple protobufs are installed and conflicts with each other, you could try solutions below:

``` shell
which protoc
# comment the output path in .bashrc if it do NOT direct to the correct protoc.
source .bashrc
sudo ldconfig
```

or

``` shell
# uninstall
sudo apt-get remove libprotobuf-dev
sudo apt-get remove protobuf-compiler
sudo apt-get remove python-protobuf
sudo rm -rf /usr/local/bin/protoc
sudo rm -rf /usr/bin/protoc
sudo rm -rf /usr/local/include/google
sudo rm -rf /usr/local/include/protobuf*
sudo rm -rf /usr/include/google
sudo rm -rf /usr/include/protobuf*

# install
sudo apt-get update
sudo ldconfig
sudo apt-get install libprotobuf* protobuf-compiler python-protobuf
```

### Cross-compile on Windows

Cross-compile on Windows is not supported currently. You may try https://github.com/microsoft/Terminal with Linux subsystem including.


### Quantized Models

We support TensorFlow Quantized Models for now. And we plan to provide a model quantizing tool based on MNN model format, which is training free.


### Unsupported Operations

``` shell
opConverter ==> MNN Converter NOT_SUPPORTED_OP: [ ANY_OP_NAME ]
```

If the MNNConverter fails with error above, one or more operations are not supported by MNN. You could submit an issue or leave a comment at pinned issue. If you want to implement it yourself, You can follow [our guide](AddOp_EN.md). Pull requests are always welcome.


__TensorFlow SSD model is not supported -- usage of TensorFlow Object API produces some unsupported control logic operations in post-processing part. And the TensorFlow SSD model is not as efficient as Caffe SSD model. So, it is recommended to use the Caffe version SSD model.__


## Runtime FAQ

### What is NC4HW4 Format ?

The difference between NCHW and NC4HW4 is just like the difference between color representing method planar and chunky. Imagine a 2x2 RGBA image, in planar representing (NCHW), its storage would be `RRRRGGGGBBBBAAAA`; and in chunky representing (NC4HW4), its storage would be `RGBARGBARGBARGBA`. In MNN, we pack each 4 channels for floats or 8 channels for int8s to gain better performance with SIMD.

You can obtain tensor's format through ```TensorUtils::getDescribe(tensor)->dimensionFormat```. If it returns `MNN_DATA_FORMAT_NC4HW4`, the channel dim is packed, which may cause tensor's elementSize be greater than product of each dimension.

### How to Convert Between Formats ?

You can convert tensor format using codes below:


``` c++
auto srcTensor = Tensor::create({1, 224, 224, 3}, Tensor::TENSORFLOW);
// ... set srcTensor data
auto dstTensor = net->getSessionInput(session, NULL);
dstTensor->copyFromHostTensor(srcTensor);
```

### Why does output tensor data copying so slow on GPU backend?

If you do not wait for GPU inference to be finished (through runSessionWithCallback with sync), copyToHostTensor has to wait for it before copying data.

