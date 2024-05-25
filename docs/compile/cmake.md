# 编译宏介绍
MNN使用CMake构建项目，CMake中的宏定义列表如下：

|          宏名         |         宏说明         |
| -------------------- | ---------------------- |
| MNN_USE_SYSTEM_LIB   | 在使用`opencl`和`vulkan`时，使用系统库（`ON`）还是通过`dlopen`引入动态库（`OFF`），默认为`OFF` |
| MNN_BUILD_HARD       | 是否使用`-mfloat-abi=hard`，默认为`OFF` |
| MNN_BUILD_SHARED_LIBS   | 是否构建为动态库，默认为`ON` |
| MNN_WIN_RUNTIME_MT      | 在Windows上构建`dll`时是否使用`/MT`，默认为`OFF` |
| MNN_FORBID_MULTI_THREAD | 是否禁止多线程，默认为`OFF` |
| MNN_OPENMP           | 是否使用`OpenMP`的线程池，该选项在`Mac/iOS`平台无效，默认为`OFF` |
| MNN_USE_THREAD_POOL  | 是否使用MNN内部线程池，默认为`ON` |
| MNN_BUILD_TRAIN      | 是否构建MNN的训练框架，默认为`OFF` |
| MNN_BUILD_DEMO       | 是否构建MNN的demo，默认为`OFF` |
| MNN_BUILD_TOOLS      | 是否构建MNN的测试工具，默认为`ON` |
| MNN_BUILD_QUANTOOLS  | 是否构建MNN的量化工具，默认为`OFF` |
| MNN_EVALUATION       | 是否构建MNN的评估工具，默认为`OFF` |
| MNN_BUILD_CONVERTER  | 是否构建MNN的转换工具，默认为`OFF` |
| MNN_SUPPORT_DEPRECATED_OP | 是否支持Tflite的量化算子，默认为`ON` |
| MNN_DEBUG_MEMORY     | 是否开启MNN内存调试，默认为`OFF` |
| MNN_DEBUG_TENSOR_SIZE | 是否开启MNN tensor size调试，默认为`OFF` |
| MNN_GPU_TRACE        | 是否开启MNN GPU调试，默认为`OFF` |
| MNN_PORTABLE_BUILD   | 尽可能链接第三方库的静态版本，以提高构建的可执行文件的可移植性，默认为`OFF` |
| MNN_SEP_BUILD        | 是否构建MNN的后端和表达式分离版本，只在`MNN_BUILD_SHARED_LIBS=ON`时生效，默认为`ON` |
| NATIVE_LIBRARY_OUTPUT | 如果构建为动态库，则指定动态库的输出路径，默认为`OFF` |
| NATIVE_INCLUDE_OUTPUT | 如果构建为动态库，则指定动态库的头文件路径，默认为`OFF` |
| MNN_AAPL_FMWK        | 是否构建`MNN.framework`替代`*.dylib`，默认为`OFF` |
| MNN_WITH_PLUGIN      | 是否支持`Plugin算子`，默认为`OFF` |
| MNN_BUILD_MINI       | 是否构建MNN的最小化版本，最小化版本仅支持固定形状，默认为`OFF` |
| MNN_USE_SSE          | 在x86上是否使用SSE指令集，默认为`OFF` |
| MNN_BUILD_CODEGEN    | 是否构建MNN的代码生成部分，该功能提供了算子融合与代码生成能力，为实验性功能，默认为`OFF` |
| MNN_ENABLE_COVERAGE  | 是否开启MNN的代码覆盖率，默认为`OFF` |
| MNN_BUILD_PROTOBUFFER | 是否使用MNN中的`protobuffer`，默认为`ON` |
| MNN_BUILD_OPENCV     | 是否构建MNN的OpenCV功能，默认为`OFF` |
| MNN_INTERNAL         | 是否构建MNN的一些内部功能，如：日志；默认为`OFF` |
| MNN_JNI              | 是否构建MNN的JNI支持，默认为`OFF` |
| MNN_METAL            | 是否构建`Metal`后端，默认为`OFF` |
| MNN_OPENCL           | 是否构建`OpenCL`后端，默认为`OFF` |
| MNN_OPENGL           | 是否构建`OpenGL`后端，默认为`OFF` |
| MNN_VULKAN           | 是否构建`Vulkan`后端，默认为`OFF` |
| MNN_ARM82            | 是否构建`Armv8.2`后端，默认为`OFF` |
| MNN_ONEDNN           | 是否使用`oneDNN`，默认为`OFF` |
| MNN_AVX512           | 是否构建`avx512`后端，默认为`OFF` |
| MNN_CUDA             | 是否构建`Cuda`后端，默认为`OFF` |
| MNN_CUDA_PROFILE     | 是否打开CUDA profile工具，默认为`OFF` |
| MNN_CUDA_QUANT       | 是否打开CUDA 量化文件编译，默认为`OFF` |
| MNN_CUDA_BF16        | 是否打开CUDA Bf16文件编译，默认为`OFF` |
| MNN_CUDA_TUNE_PARAM  | 是否打开CUDA TUNE相关文件编译，目前仅支持安培及以上架构，默认为`OFF` |
| MNN_TENSORRT         | 是否构建`TensorRT`后端，默认为`OFF` |
| MNN_COREML           | 是否构建`CoreML`后端，默认为`OFF` |
| MNN_NNAPI            | 是否构建`NNAPI`后端，默认为`OFF`  |
| MNN_BUILD_BENCHMARK  | 是否构建MNN的性能测试，默认为`OFF` |
| MNN_BUILD_TEST       | 是否构建MNN的单元测试，默认为`OFF` |
| MNN_BUILD_FOR_ANDROID_COMMAND | 是否使用命令行构建`Android`，默认为`OFF` |
| MNN_USE_LOGCAT       | 是否使用`logcat`代替`printf`输出日志，默认为`OFF` |
| MNN_USE_CPP11        | 是否使用`C++11`编译MNN，默认为`ON` |
| MNN_SUPPORT_BF16     | 是否支持`BF16`，默认为`OFF` |
| MNN_SSE_USE_FP16_INSTEAD | 在X86平台是否使用`FP16`替代`BF16`，默认为`OFF` |
| MNN_AVX512_VNNI      | 是否使用`avx512_vnni`指令，该宏仅在`MNN_AVX512=ON`时生效，默认为`OFF` |
| MNN_OPENCL_SIZE_CUT  | 是否为了降低OpenCL大小而关闭OpenCL Buffer实现，该宏仅在`MNN_OPENCL=ON`时生效，默认为`OFF` |
| MNN_OPENCL_PROFILE   | 是否打开OpenCL Kernel性能Profile，该宏仅在`MNN_OPENCL=ON`时生效，默认为`OFF` |
| MNN_METALLIB_SOURCE  | 使用Metal时是否直接使用Metal源码，该宏仅在`MNN_METAL=ON`时生效，默认为`ON` |
| MNN_VULKAN_DEBUG     | 是否打开Vulkan的DEBUG模式，该宏仅在`MNN_VULKAN=ON`时生效，默认为`OFF` |
| MNN_OPENGL_REGEN     | 是否重新生成OpenGL Kenel，该宏仅在`MNN_OPENGL=ON`时生效，默认为`OFF` |
| MNN_TRT_DYNAMIC      | 是否通过dlopen的方式引入TRT的动态库，该宏仅在`MNN_TENSORRT=ON`时生效，默认为`OFF |
| MNN_BUILD_TORCH      | 构建的`MNNConvert`是否支持`TorchScript`，该宏仅在`MNN_BUILD_CONVERTER=ON`时生效，默认为`OFF` |
| MNN_TRAIN_DEBUG      | 构建的训练模块是否支持调试，该宏仅在`MNN_BUILD_TRAIN=ON`时生效，默认为`OFF` |
| MNN_USE_OPENCV       | 构建的训练Demo是否使用`OpenCV`依赖，该宏仅在`MNN_BUILD_TRAIN=ON`时生效，默认为`OFF` |
| MNN_IMGPROC_COLOR    | 构建MNN的OpenCV功能是否开启`颜色空间转换`，默认为`ON` |
| MNN_IMGPROC_GEOMETRIC | 构建MNN的OpenCV功能是否开启`形变`，默认为`ON` |
| MNN_IMGPROC_DRAW     | 构建MNN的OpenCV功能是否开启`画图`，默认为`ON` |
| MNN_IMGPROC_FILTER   | 构建MNN的OpenCV功能是否开启`滤波`，默认为`ON` |
| MNN_IMGPROC_MISCELLANEOUS | 构建MNN的OpenCV功能是否开启`混合`，默认为`ON` |
| MNN_IMGPROC_STRUCTRAL | 构建MNN的OpenCV功能是否开启`结构`，默认为`ON` |
| MNN_IMGPROC_HISTOGRAMS | 构建MNN的OpenCV功能是否开启`直方图`，默认为`ON` |
| MNN_CALIB3D          | 构建MNN的OpenCV功能是否开启`3d`，默认为`ON` |
| MNN_IMGCODECS        | 构建MNN的OpenCV功能是否开启`图像编解码`，默认为`OFF` |
| MNN_CVCORE           | 构建MNN的OpenCV功能是否开启`core`功能，默认为`ON` |
| MNN_OPENCV_TEST      | 构建MNN的OpenCV功能是否开启单元测试，默认为`OFF` |
| MNN_OPENCV_BENCH     | 构建MNN的OpenCV功能是否开启性能benchmark，默认为`OFF` |
| MNN_VULKAN_IMAGE     | 构建MNN的Vulkan后端时采用Image内存模式，以便支持FP16和部分移动端上GPU的加速，默认为`ON` |
| MNN_LOW_MEMORY       | 是否支持低内存模式，支持低内存模式使用权值量化模型并设置`low_memory`则会使用计算时反量化，默认为`OFF` |
| MNN_SUPPORT_RENDER   | 是否支持图形渲染相关算子实现，默认为 `OFF` |
| MNN_SUPPORT_TRANSFORMER_FUSE | 是否支持Fuse Transformer相关OP实现，默认为 `OFF` |
| MNN_BUILD_LLM        | 是否构建基于MNN的llm库和demo，默认为`OFF` |
| MNN_BUILD_DIFFUSION  | 是否构建基于MNN的diffusion demo，需要打开MNN_BUILD_OPENCV和MNN_IMGCODECS宏使用 默认为`OFF` |
