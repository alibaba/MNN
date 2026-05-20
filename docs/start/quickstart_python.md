# Python 快速开始（5分钟）

本教程带你从安装到推理，完成一个端到端的图像分类任务。

## 1. 安装 MNN

```bash
pip install MNN
```

> 如果 pip 安装失败（可能是当前系统和 Python 版本不支持），可以从源码编译安装，参考 [PyMNN 编译](../compile/pymnn.md)。

## 2. 准备模型

以 MobileNet V1 为例，先将 ONNX 模型转换为 MNN 格式：

```bash
# 转换为 MNN 格式（附带 8bit 权值量化，体积缩小 75%）
mnnconvert -f ONNX --modelFile mobilenet_v1.onnx --MNNModel mobilenet_v1.mnn --weightQuantBits 8
```

> 更多转换选项请参考 [模型转换工具](../tools/convert.md) 和 [模型压缩](../tools/compress.md)。

## 3. 加载模型并推理

```python
from __future__ import print_function
import numpy as np
import MNN
import MNN.cv as cv2
import sys

def inference(model_path, image_path):
    # ========== 1. 加载模型 ==========
    # 参数：模型路径、输入名列表、输出名列表
    net = MNN.nn.load_module_from_file(model_path, ["input"], ["MobilenetV1/Predictions/Reshape_1"])

    # ========== 2. 预处理 ==========
    image = cv2.imread(image_path)
    image = image[..., ::-1]              # BGR -> RGB
    image = cv2.resize(image, (224, 224))
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    image = image.astype(np.float32)

    # 创建输入 VARP：[N, H, W, C] NHWC 格式
    input_var = MNN.expr.placeholder([1, 224, 224, 3], MNN.expr.NHWC)
    input_var.write(image)

    # Module 内部使用 NC4HW4 格式，需要转换
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)

    # ========== 3. 推理 ==========
    output_var = net.forward([input_var])
    output_var = output_var[0]

    # ========== 4. 后处理 ==========
    # 输出也可能是 NC4HW4，转回 NHWC 再读取
    output_var = MNN.expr.convert(output_var, MNN.expr.NHWC)
    print("预测类别编号：{}".format(np.argmax(output_var.read())))

if __name__ == "__main__":
    inference(sys.argv[1], sys.argv[2])
```

运行：
```bash
python classify.py mobilenet_v1.mnn test.jpg
```

## 4. 使用 GPU 加速（可选）

```python
config = {
    'backend': 3,       # OpenCL GPU
    'precision': 2,     # FP16
    'numThread': 4,
}
rt = MNN.nn.create_runtime_manager((config,))
rt.set_cache("gpu.cache")

net = MNN.nn.load_module_from_file("mobilenet_v1.mnn",
                                    ["input"],
                                    ["MobilenetV1/Predictions/Reshape_1"],
                                    runtime_manager=rt)
```

> 后端选项：0=CPU, 1=Metal, 2=CUDA, 3=OpenCL, 7=Vulkan。详见 [PyMNN 完整指南](python.md)。

## 下一步

- [LLM 部署指南](../transformers/llm.md) — 部署大语言模型
- [模型压缩](../tools/compress.md) — 权值量化、离线量化、FP16 压缩
- [C++ 推理-Module API](../inference/module.md) — C++ 高性能推理接口
- [Python API 参考](../pymnn/MNN.md) — 完整 Python API 文档
