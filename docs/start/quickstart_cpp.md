# C++ 快速开始

本教程带你从源码编译到 C++ 推理，完成一个端到端的图像分类任务。

## 1. 编译 MNN

```bash
cd MNN
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j8
```

> 更多平台（iOS/Android/Windows）编译方式请参考 [从源码构建](../compile/engine.md)。

## 2. 转换模型

```bash
./MNNConvert -f ONNX --modelFile mobilenet_v1.onnx --MNNModel mobilenet_v1.mnn --weightQuantBits 8
```

## 3. 编写推理代码

使用推荐的 **Module API** 进行推理（参考 `demo/exec/pictureRecognition_module.cpp`）：

```cpp
#include <stdio.h>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/ImageProcess.hpp>

using namespace MNN;
using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: ./classify model.mnn input.jpg\n");
        return 1;
    }

    // 1. 创建 RuntimeManager 配置后端
    ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_CPU;
    sConfig.numThread = 4;
    auto rtmgr = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sConfig));

    // 2. 加载模型（空 vector 表示自动检测输入输出名）
    std::shared_ptr<Module> net(
        Module::load({}, {}, argv[1], rtmgr));
    if (!net) {
        printf("Failed to load model\n");
        return 1;
    }

    // 3. 创建输入 (NC4HW4 是 MNN 内部优化格式)
    int width = 224, height = 224;
    auto input = _Input({1, 3, height, width}, NC4HW4);

    // 4. 图像预处理（使用 MNN ImageProcess）
    // 实际项目中用 stb_image/opencv 读图后用 ImageProcess 转换
    // 此处简化为填充测试数据
    auto inputPtr = input->writeMap<float>();
    // ... 填充预处理后的图像数据到 inputPtr ...
    input->unMap();

    // 5. 推理
    auto outputs = net->onForward({input});

    // 6. 读取输出
    auto output = _Convert(outputs[0], NHWC);
    output = _Reshape(output, {0, -1});
    auto outputPtr = output->readMap<float>();
    // 找到概率最大的类别
    int classId = 0;
    float maxVal = outputPtr[0];
    int outputSize = output->getInfo()->size;
    for (int i = 1; i < outputSize; i++) {
        if (outputPtr[i] > maxVal) {
            maxVal = outputPtr[i];
            classId = i;
        }
    }
    printf("预测类别编号: %d, 置信度: %f\n", classId, maxVal);
    return 0;
}
```

## 4. 编译并运行

在 MNN build 目录下：
```bash
g++ -std=c++11 -o classify classify.cpp \
    -I ../include \
    -L . -lMNN -lMNN_Express

./classify mobilenet_v1.mnn test.jpg
```

## 5. 完整的图像处理示例

实际项目中使用 `ImageProcess` 完成图像解码和预处理，参考 `demo/exec/pictureRecognition_module.cpp`，核心流程：

```cpp
#include <MNN/ImageProcess.hpp>

// 配置预处理参数
MNN::CV::ImageProcess::Config imgConfig;
imgConfig.filterType = MNN::CV::BILINEAR;
imgConfig.sourceFormat = MNN::CV::RGBA;
imgConfig.destFormat = MNN::CV::RGB;
imgConfig.mean[0] = 103.94f;
imgConfig.mean[1] = 116.78f;
imgConfig.mean[2] = 123.68f;
imgConfig.normal[0] = 0.017f;
imgConfig.normal[1] = 0.017f;
imgConfig.normal[2] = 0.017f;

auto pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
    MNN::CV::ImageProcess::create(imgConfig));

// 设置缩放矩阵
MNN::CV::Matrix trans;
trans.setScale((float)(inputWidth - 1) / (width - 1),
               (float)(inputHeight - 1) / (height - 1));
pretreat->setMatrix(trans);

// 转换图像数据到 input VARP
pretreat->convert(imageData, inputWidth, inputHeight, 0,
                  input->writeMap<float>(), width, height, 4, 0,
                  halide_type_of<float>());
```

## 下一步

- [Module API 详解](../inference/module.md) — 完整的 C++ 推理接口说明
- [LLM 部署指南](../transformers/llm.md) — 部署大语言模型
- [模型压缩](../tools/compress.md) — 权值量化、离线量化等
- [Session API](../inference/session.md) — 低层推理接口（特殊场景使用）
