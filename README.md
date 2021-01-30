![MNN](doc/banner.png)

[中文版本](README_CN.md)

[MNN Homepage](http://www.mnn.zone)

## Intro
MNN is a highly efficient and lightweight deep learning framework. It supports inference and training of deep learning models, and has industry leading performance for inference and training on-device. At present, MNN has been integrated in more than 20 apps of Alibaba Inc, such as Taobao, Tmall, Youku, Dingtalk, Xianyu and etc., covering more than 70 usage scenarios such as live broadcast, short video capture, search recommendation, product searching by image, interactive marketing, equity distribution, security risk control. In addition, MNN is also used on embedded devices, such as IoT.

The design principles and performance data of MNN has been published in an MLSys 2020 paper [here](https://proceedings.mlsys.org/static/paper_files/mlsys/2020/7-Paper.pdf). Please cite MNN in your publications if it helps your research:

    @inproceedings{alibaba2020mnn,
      author = {Jiang, Xiaotang and Wang, Huan and Chen, Yiliu and Wu, Ziqi and Wang, Lichuan and Zou, Bin and Yang, Yafeng and Cui, Zongyang and Cai, Yu and Yu, Tianhang and Lv, Chengfei and Wu, Zhihua},
      title = {MNN: A Universal and Efficient Inference Engine},
      booktitle = {MLSys},
      year = {2020}
    }

## Documentation and Tools
MNN's docs are in placed in [Yuque docs here](https://www.yuque.com/mnn/en).

MNN Workbench could be downloaded from [MNN's homepage](http://www.mnn.zone), which provides pretrained models, visualized training tools, and one-click deployment of models to devices.

## Key Features
### High performance
- Implements core computing with lots of optimized assembly code to make full use of the ARM CPU.
- For iOS, GPU acceleration (Metal) can be turned on, which is faster than Apple's native CoreML.
- For Android, `OpenCL`, `Vulkan`, and `OpenGL` are available and deep tuned for mainstream GPUs (`Adreno` and `Mali`).
- Convolution and transposition convolution algorithms are efficient and stable. The Winograd convolution algorithm is widely used to better symmetric convolutions such as 3x3 -> 7x7.
- Twice speed increase for the new architecture ARM v8.2 with FP16 half-precision calculation support.

### Lightweight
- Optimized for devices, no dependencies, can be easily deployed to mobile devices and a variety of embedded devices.
- iOS platform: static library size for armv7+arm64 platforms is about 5MB, size increase of linked executables is about 620KB, and metallib file is about 600KB.
- Android platform: core so size is about 400KB, OpenCL so is about 400KB, Vulkan so is about 400KB.

### Versatility
- Supports `Tensorflow`, `Caffe`, `ONNX`, and supports common neural networks such as `CNN`, `RNN`, `GAN`.
- MNN model converter supports 149 `Tensorflow` OPs, 58 `TFLite` OPs, 47 `Caffe` OPs and 74 `ONNX` OPs; Number of OPs by different MNN hardware backends: 111 for CPU, 6 for ARM V8.2, 55 for Metal, 43 for OpenCL, and 32 for Vulkan.
- Supports iOS 8.0+, Android 4.3+ and embedded devices with POSIX interface.
- Supports hybrid computing on multiple devices. Currently supports CPU and GPU.

### Ease of use
- Efficient image processing module, speeding up affine transform and color space transform without libyuv or opencv.
- Provides callbacks throughout the workflow to extract data or control the execution precisely.
- Provides options for selecting inference branch and paralleling branches on CPU and GPU.
- (BETA) MNN Python API helps ML engineers to easily use MNN to build a model, train it and quantize it, without dipping their toes in C++ code.

## Architecture
![architecture](doc/architecture.png)

MNN can be divided into two parts: Converter and Interpreter.

Converter consists of Frontends and Graph Optimize. The former is responsible for supporting different training frameworks. MNN currently supports Tensorflow, Tensorflow Lite, Caffe and ONNX (PyTorch/MXNet); the latter optimizes graphs by operator fusion, operator substitution, and layout adjustment.

Interpreter consists of Engine and Backends. The former is responsible for the loading of the model and the scheduling of the calculation graph; the latter includes the memory allocation and the Op implementation under each computing device. In Engine and Backends, MNN applies a variety of optimization schemes, including applying Winograd algorithm in convolution and deconvolution, applying Strassen algorithm in matrix multiplication, low-precision calculation, Neon optimization, hand-written assembly, multi-thread optimization, memory reuse, heterogeneous computing, etc.

## How to Discuss and Get Help From MNN Community

Scan the following QR codes to join Dingtalk discussion group. The group discussions are predominantly Chinese. But we welcome and will help English speakers.

Group #1 (Full):

<img src="doc/DingTalkQR1.png" height="256"/>

Group #2 (Full):

<img src="doc/DingTalkQR2.png" height="256"/>

Group #3:

<img src="doc/DingTalkQR3.png" height="256"/>

## License
Apache 2.0

## Acknowledgement
MNN participants: Taobao Technology Department, Search Engineering Team, DAMO Team, Youku and other Alibaba Group employees.

MNN refers to the following projects:
- [Caffe](https://github.com/BVLC/caffe)
- [flatbuffer](https://github.com/google/flatbuffers)
- [gemmlowp](https://github.com/google/gemmlowp)
- [Google Vulkan demo](http://www.github.com/googlesamples/android-vulkan-tutorials)
- [Halide](https://github.com/halide/Halide)
- [Mace](https://github.com/XiaoMi/mace)
- [ONNX](https://github.com/onnx/onnx)
- [protobuffer](https://github.com/protocolbuffers/protobuf)
- [skia](https://github.com/google/skia)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [ncnn](https://github.com/Tencent/ncnn)
- [paddle-mobile](https://github.com/PaddlePaddle/paddle-mobile)
- [stb](https://github.com/nothings/stb)
- [rapidjson](https://github.com/Tencent/rapidjson)
- [pybind11](https://github.com/pybind/pybind11)
- [pytorch](https://github.com/pytorch/pytorch)
- [bolt](https://github.com/huawei-noah/bolt)
- [libyuv](https://chromium.googlesource.com/libyuv/libyuv)
