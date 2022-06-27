![MNN](doc/banner.png)

[中文版本](README_CN.md)

[MNN Homepage](http://www.mnn.zone)

## Intro
MNN is a highly efficient and lightweight deep learning framework. It supports inference and training of deep learning models, and has industry leading performance for inference and training on-device. At present, MNN has been integrated in more than 30 apps of Alibaba Inc, such as Taobao, Tmall, Youku, Dingtalk, Xianyu and etc., covering more than 70 usage scenarios such as live broadcast, short video capture, search recommendation, product searching by image, interactive marketing, equity distribution, security risk control. In addition, MNN is also used on embedded devices, such as IoT.

The design principles and performance data of MNN has been published in an MLSys 2020 paper [here](https://arxiv.org/pdf/2002.12418.pdf). Please cite MNN in your publications if it helps your research:

    @inproceedings{alibaba2020mnn,
      author = {Jiang, Xiaotang and Wang, Huan and Chen, Yiliu and Wu, Ziqi and Wang, Lichuan and Zou, Bin and Yang, Yafeng and Cui, Zongyang and Cai, Yu and Yu, Tianhang and Lv, Chengfei and Wu, Zhihua},
      title = {MNN: A Universal and Efficient Inference Engine},
      booktitle = {MLSys},
      year = {2020}
    }

![image.png](doc/workflow.png)

## Documentation and Workbench
MNN's docs are in placed in [Yuque docs here](https://www.yuque.com/mnn/en).

MNN Workbench could be downloaded from [MNN's homepage](http://www.mnn.zone), which provides pretrained models, visualized training tools, and one-click deployment of models to devices.

## Key Features
### Lightweight
- Optimized for devices, no dependencies, can be easily deployed to mobile devices and a variety of embedded devices.
- iOS platform: static library size will full option for armv7+arm64 platforms is about 12MB, size increase of linked executables is about 2M.
- Android platform: core so size is about 800KB (armv7a - c++_shared).
- Use MNN_BUILD_MINI can reduce package size about 25% , with limit of fix model input size
- Support FP16 / Int8 qunatize, can reduce model size 50%-70%

### Versatility
- Supports `Tensorflow`, `Caffe`, `ONNX`,`Torchscripts` and supports common neural networks such as `CNN`, `RNN`, `GAN`, `Transformork`.
- Supports AI model with multi-inputs or multi-outputs, every kind of dimenstion format, dynamic inputs, controlflow.
- MNN supports approximate full OPs used for AI Model. The converter supports 178 `Tensorflow` OPs, 52 `Caffe` OPs, 163 `Torchscripts` OPs, 158 `ONNX` OPs.
- Supports iOS 8.0+, Android 4.3+ and embedded devices with POSIX interface.
- Supports hybrid computing on multiple devices. Currently supports CPU and GPU.


### High performance
- Implements core computing with lots of optimized assembly code to make full use of the ARM / x64 CPU.
- Use Metal / OpenCL / Vulkan to support GPU inference on mobile.
- Use CUDA and tensorcore to support NVIDIA GPU for better performance
- Convolution and transposition convolution algorithms are efficient and stable. The Winograd convolution algorithm is widely used to better symmetric convolutions such as 3x3,4x4,5x5,6x6,7x7.
- Twice speed increase for the new architecture ARM v8.2 with FP16 half-precision calculation support. 2.5 faster to use sdot for ARM v8.2 and VNNI.

### Ease of use
- Support use MNN's OP to do numerical calculating like numpy.
- Support lightweight image process module like OpenCV, which is only 100k.
- Support build model and train it on PC / mobile.
- MNN Python API helps ML engineers to easily use MNN to inference, train, process image, without dipping their toes in C++ code.


- S ：Support and work well, deeply optimized, recommend to use
- A ：Support and work well, can use
- B ：Support but has bug or not optimized, no recommend to use
- C ：Not Support

| Architecture / Precision |  | Normal | FP16 | BF16 | Int8 |
| --- | --- | --- | --- | --- | --- |
| CPU | Native | B | C | B | B |
|  | x86/x64-SSE4.1 | A | B | B | A |
|  | x86/x64-AVX2 | S | B | B | A |
|  | x86/x64-AVX512 | S | B | B | S |
|  | ARMv7a | S | S (ARMv8.2) | S | S |
|  | ARMv8 | S | S (ARMv8.2) | S | S |
| GPU | OpenCL | A | S | C | C |
|  | Vulkan | A | A | C | C |
|  | Metal | A | S | C | C |
|  | CUDA | A | S | C | C |
| NPU | CoreML | B | C | C | C |
|  | HIAI | B | C | C | B |



## Architecture
![architecture](doc/architecture.png)

MNN can be divided into two parts: Inference Engine and Tools.

### Inference Engine

The input of Inference Engine, AI model is a Directed Acyclic Graph(DAG), each node in model is an operator, which describe a kind of tensor compute function. Inference Engine will load and execute the graph. It can seperate into schedule and execute:
![runflow.png](doc/runflow.png)

- Schedule: Load Graph and Pretreat it
    - Decompose OP, reduce kinds of OPs
    - Search best compute stratagy
    - Find best resource allocation
- Execute: Implete OP, use algorithm and hardware feature to optimize
    - Algorithm: Winograd Convolution, Strassen Matrix Multiply, Low Precision Compute
    - Hardware: SIMD for CPU (SSE/NEON/AVX), GPU API (OpenCL / CUDA / Metal)

### Tools
- MNN-Converter: Convert other model to MNN model, such as Tensorflow(lite), Caffe, ONNX, Torchscripts. And do graph optimization to reduce computation.
- MNN-Compress: Compress model to reduce size and increase performance / speed
- MNN-Express: Support model with controlflow, use MNN's OP to do general-purpose compute.
- MNN-CV: A OpenCV liked library, but based on MNN and then much more lightweight.
- MNN-Train: Support train MNN model.

## How to Discuss and Get Help From MNN Community

The group discussions are predominantly Chinese. But we welcome and will help English speakers.

Dingtalk discussion groups:

Group #1 (Full): 23329087

Group #2 (Full): 23350225

Group #3: https://h5.dingtalk.com/circle/healthCheckin.html?dtaction=os&corpId=ding8989a1d6ae6ef130b177420cc0e366ea&f0c81=1b93a&cbdbhh=qwertyuiop

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
- [libjpeg](https://github.com/libjpeg-turbo/libjpeg-turbo)
- [opencv](https://github.com/opencv/opencv)
