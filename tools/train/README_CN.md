# MNN 训练相关工具

## 编译
### MNN 编译与安装
- MNN 编译时打开 MNN_BUILD_TRAIN 开关：cmake .. -DMNN_BUILD_TRAIN=true

### 产物
- transformer
- extractForInfer
- runTrainDemo.out


## 使用
### 制作训练模型
eg: ./transformer mobilenet.mnn mobilenet-train.mnn transformerConfig.json [revert.json]

- 第一个参数：输入，推理模型或计算Loss的模型
- 第二个参数：输出，训练模型
- 第三个参数：输入，为json配置文件, 参考 transformerConfig.json 和 transformerConfig2.json 编写
- 第四个参数：输出，还原参数所需要的配置文件，若不指定，默认为当前路径下的 revert.json

### 制作训练数据
- 根据模型输入和标注准备

### 训练
- 用 Interpreter-Session API 加载训练模型，得到 Session
- 根据 transformer 打印的输入，传入 Session
- 运行 Session
- （可选）获取 Session 中的 loss 并查看
- 若干次迭代后，调用 Interpreter 的 updateSessionToModel 更新权重, 然后调用 getModelBuffer 获取新的模型内存并自行写入到新的模型文件中

### 应用
- 使用 extractForInfer 将训练模型的参数提取到 推理模型
- 重新加载推理模型，即训练好的模型
