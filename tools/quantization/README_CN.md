# 模型压缩与量化
## 量化的作用
量化将网络中主要算子（卷积）由原先的浮点计算转成低精度的Int8计算，减少模型大小并提升性能

## 编译
### 编译宏
编译MNN时开启 MNN_BUILD_QUANTOOLS 宏，即开启量化工具的编译

### 编译产物
量化模型的工具： quantized.out
量化模型与浮点模型的对比工具：testQuanModel.out

## 量化工具的使用
### 命令
```bash
./quantized.out origin.mnn quan.mnn pretreatConfig.json
```

第一个参数为原始模型文件路径，即待量化的浮点模型

第二个参数为目标模型文件路径，即量化后的模型

第三个参数为预处理的配置项，参考 mobilenetCaffeConfig.json

### Json 配置
#### format
可选："RGB", "BGR", "RGBA", "GRAY"

#### mean, normal
同 ImageProcess 的配置

dst = (src - mean) * normal

#### width, height
模型输入的宽高

#### path
图片目录，一般放2-10张图片即可

## 量化模型的使用
和浮点模型同样使用方法，输入输出仍然为浮点类型
