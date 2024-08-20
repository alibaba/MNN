# 其他模块编译

## 模型转换工具
- 相关编译选项
  - `MNN_BUILD_CONVERTER` 是否编译模型转换工具
  - `MNN_BUILD_TORCH` 是否支持TorchScript模型转换，MacOS下需要安装pytorch，Linux下会下载libtorch
- 编译命令
    ```bash
    cmake .. -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TORCH=ON
    ```
- 编译产物
  - `MNNConvert` 模型转换工具
  - `TestConvertResult` 模型转换正确性测试工具，*Windows下没有此产物，用`MNNConvert`对应功能替代*
  - `TestPassManager` 模型转换工具测试用例
  - `MNNDump2Json` 模型转换为Json
  - `MNNRevert2Buffer` Json转换为模型
  - `OnnxClip` Onnx模型裁剪工具
## 训练框架
- 相关编译选项
  - `MNN_BUILD_TRAIN` 是否编译训练框架
  - `MNN_BUILD_TRAIN_MINI` 对于移动端/嵌入式设备，建议设置`MNN_BUILD_TRAIN_MINI=ON`，不编译内置的`Dataset`，`Models`，这部分在移动端/嵌入式设备上一般有其他解决方案
  - `MNN_USE_OPENCV` 部分PC上的demo有用到，与Dataset处理相关
- 编译命令
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_TRAIN=ON -DMNN_USE_OPENCV=ON
    make -j4
    ```
- 编译产物
  - `MNNTrain` 训练框架库
  - `runTrainDemo.out` 运行训练框架demo的入口程序
  - `transformer` 训练模型转换器，将推理用的MNN模型转换为执行训练的MNN模型
  - `extractForInfer` 从执行训练的MNN模型中提取参数，对应更新推理用的MNN模型
## 生成式模型
- 相关编译选项
  - `MNN_BUILD_DIFFUSION` 是否编译扩散模型推理示例
  - `MNN_BUILD_LLM` 是否编译大语言模型推理引擎
  - `MNN_SUPPORT_TRANSFORMER_FUSE` 是否支持`transformer`相关的融合算子，主要加速transformer模型
- 编译命令
  - 编译扩散模型推理示例
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_BUILD_DIFFUSION=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
    make -j4
    ```
  - 编译大语言模型推理引擎
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
    make -j4
    ```
- 编译产物
  - `libllm.so` 大语言模型推理库
  - `llm_demo` 大语言模型推理示例程序
  - `diffusion_demo` 扩散模型示例程序
## 测试工具
- 相关编译选项
  - `MNN_BUILD_TOOL` 是否编译测试工具
- 编译命令
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_TOOL=ON
    make -j4
    ```
- 编译产物
  - `GetMNNInfo` 获取MNN模型信息
  - `ModuleBasic.out` 使用V3 API对模型执行基础推理测试
  - `SequenceModuleTest.out` 测试Sequence模型推理
  - `MNNV2Basic.out` 使用V2 API对模型执行基础推理测试
  - `mobilenetTest.out` 测试mobilenet模型推理
  - `backendTest.out` 测试模型在指定后端上执行的结果是否与CPU一致
  - `modelCompare.out` 原始模型与量化模型推理结果比较
  - `testModel.out` 给定输入输出测试模型推理正确性
  - `testModel_expr.out` 给定输入输出测试模型推理正确性
  - `testModelWithDescribe.out` 给定输入输出和shape描述测试模型推理正确性
  - `getPerformance.out`  获取当前设备的CPU性能
  - `checkInvalidValue.out` 检测输出目录里的数据
  - `timeProfile.out` 测试模型在指定后端上执行的时间，并获取每层的执行时间占比
  - `testTrain.out` 测试训练功能
  - `checkDir.out`  测试两个文件夹是否一致
  - `checkFile.out` 测试两个文件是否一致
  - `winogradExample.out` winograd示例
  - `fuseTest` 测试 GPU 自定义算子的功能，目前仅支持 Vulkan Buffer 模式
  - `GpuInterTest.out` 测试 GPU 内存输入的功能，目前仅支持 OpenCL Buffer 模式与 OpenGL texture 模式，编译时许打开 MNN_OPENCL 与 MNN_OPENGL
  - `LoRA` 将LorA权重添加到模型权重中
## Benchmark工具
- 相关编译选项
  - `MNN_BUILD_BENCHMARK` 是否编译Benchmark工具
- 编译命令
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_BENCHMARK=ON
    make -j4
    ```
- 编译产物
  - `benchmark.out` benchmark工具
  - `benchmarkExprModels.out` 表达式构图模型测试benchmark工具
## 模型量化工具
- 相关编译选项
  - `MNN_BUILD_QUANTOOLS` 是否编译模型量化工具
- 编译命令
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_QUANTOOLS=ON
    make -j4
    ```
- 编译产物
  - `quantized.out` 模型量化工具
## 评估工具
- 相关编译选项
  - `MNN_EVALUATION` 是否编译图片分类结果评估工具
- 编译命令
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_EVALUATION=ON
    make -j4
    ```
- 编译产物
  - `classficationTopkEval.out` 图片分类结果评估工具
## MNN OpenCV库
- 相关编译选项
  - `MNN_BUILD_OPENCV` 是否编译OpenCV函数接口
  - `MNN_IMGCODECS` 是否编译OpenCV图像解码器
  - `MNN_OPENCV_TEST` 是否编译OpenCV单元测试
  - `MNN_OPENCV_BENCH` 是否编译OpenCV性能测试
- 编译命令
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_OPENCV_TEST=ON -DMNN_OPENCV_BENCH=ON
    make -j4
    ```
- 编译产物
  - `libMNNOpenCV.so` MNN OpenCV函数库
  - `opencv_test` MNN OpenCV单元测试
  - `opencv_bench` MNN OpenCV性能测试

## 示例工程
- 相关编译选项
  - `MNN_BUILD_DEMO` 是否编译MNN Demo
- 编译命令
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_DEMO=ON
    make -j4
    ```
- 编译产物
  - `pictureRecognition.out` V2接口(Session)图片识别示例
  - `pictureRecognition_module.out` V3接口(Module)图片识别示例
   - `pictureRecognition_batch.out` 自定义batchsize图片识别示例
  - `multithread_imgrecog.out` 多线程图片识别示例
  - `pictureRotate.out` 图片旋转示例
  - `multiPose.out` 姿态检测示例
  - `segment.out` 图像实例分割示例
  - `expressDemo.out` 表达式接口推理示例
  - `expressMakeModel.out` 使用表达式构建模型示例
  - `transformerDemo.out` Transformer模型示例
  - `rasterDemo.out` Raster示例
  - `nluDemo.out` nlu模型示例
  - `mergeInplaceForCPU` 将模型中可以Inplace计算的算子改成Inplace计算，可以减少内存占用，但限定CPU后端运行
## 单元测试
- 相关编译选项
  - `MNN_BUILD_TEST` 是否编译MNN单元测试
- 编译命令
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_TEST=ON
    make -j4
    ```
- 编译产物
  - `run_test.out` 单元测试程序
