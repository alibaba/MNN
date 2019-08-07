# 模型压缩与量化
## 量化的作用
量化将网络中主要算子（卷积）由原先的浮点计算转成低精度的Int8计算，减少模型大小并提升性能

## 编译
### 编译宏
编译MNN时开启`MNN_BUILD_QUANTOOLS`宏，即开启量化工具的编译

### 编译产物
量化模型的工具： `quantized.out`
量化模型与浮点模型的对比工具：`testQuanModel.out`

## 量化工具的使用
### 命令
```bash
./quantized.out origin.mnn quan.mnn pretreatConfig.json
```

第一个参数为原始模型文件路径，即待量化的浮点模型<br>
第二个参数为目标模型文件路径，即量化后的模型<br>
第三个参数为预处理的配置项，参考 mobilenetCaffeConfig.json，mobilenetTfConfig.json

### Json 配置
#### format
图片统一按RGBA读取，然后转换到`format`指定格式<br>
可选："RGB", "BGR", "RGBA", "GRAY"

#### mean, normal
同 ImageProcess 的配置<br>
dst = (src - mean) * normal

#### width, height
模型输入的宽高

#### path
存放校正特征量化系数的图片目录

#### used_image_num
用于指定使用上述目录下多少张图片进行校正，默认使用`path`下全部图片

*注意：请确保图片经过上述步骤处理之后的数据是输入到模型input接口的数据*

#### feature_quantize_method
指定计算特征量化系数的方法<br>
可选：
1. "KL": 使用KL散度进行特征量化系数的校正，一般需要100 ~ 1000张图片
2. "ADMM": 使用ADMM（Alternating Direction Method of Multipliers）方法进行特征量化系数的校正，一般需要一个batch的数据

默认："KL"

#### weight_quantize_method
指定权值量化方法<br>
可选：
1. "MAX_ABS": 使用权值的绝对值的最大值进行对称量化
2. "ADMM": 使用ADMM方法进行权值量化

默认："MAX_ABS"

上述特征量化方法和权值量化方法可进行多次测试，择优使用

## 量化模型的使用
和浮点模型同样使用方法，输入输出仍然为浮点类型
