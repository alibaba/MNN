# MNN 训练相关工具

## 编译
### MNN 编译与安装
- MNN 编译时打开 MNN_SUPPORT_TRAIN 开关：cmake .. -DMNN_SUPPORT_TRAIN=true

### 产物
- transformer.out
- rawDataTransform.out
- dataTransformer.out
- train.out
- backendTest.out
- backwardTest.out


## 使用
### 制作训练模型
eg: ./transformer.out mobilenet.mnn mobilenet-train.mnn transformerConfig.json

- 第一个参数为推理模型
- 第二个参数为产出物训练模型
- 第三个参数为json配置文件, 参考 transformerConfig.json 和 transformerConfig2.json 编写

### 制作训练数据
#### 基于图像数据制作
eg: ./dataTransformer.out mobilenet.mnn filePath.json testData.bin

- 第一个参数为推理模型
- 第二个参数为图片描述，参考 filePath.json
- 第三个参数为产出物训练数据

#### 基于文本形式制作
eg: ./rawDataTransform.out dataConfig.json data.bin

- 第一个参数为配置文件，参考 dataConfig.json 编写
- 第二个参数为产出物训练数据

### 训练
eg: ./train.out mobilenet-train.mnn testData.bin 1000 0.01 32 Loss

- 默认程序运行完成后输出模型文件 trainResult.bin
- 第一个参数为训练模型
- 第二个参数为训练数据
- 第三个参数为迭代次数
- 第四个参数为学习率
- 第五个参数为Batch size
- 第六个参数为 Loss 函数名，若不输视为 "Loss"


## 目前支持转换的模型
- MobilenetV2: ../../AliNNModel/MobileNetV2/mobilenet_v2_1.0_224.tflite.alinn
- Lenet

注：Caffe 的 mobilenet 产生的 MNN 模型，在转成训练模型过程中会出现反卷积与卷积维度不一的情况，待解决
