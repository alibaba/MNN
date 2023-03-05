# 单输入模型离线量化工具
`./quantized.out origin.mnn quan.mnn imageInputConfig.json`

通用（任意输入个数、维度、类型）模型离线量化请看[说明](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html#id10)

MNN现已推出基于TensorFlow/Pytorch的模型压缩工具mnncompress，请查看[文档](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html)选择使用

## 参数
  - 第一个参数为原始模型文件路径，即待量化的浮点模
  - 第二个参数为目标模型文件路径，即量化后的模型
  - 第三个参数为预处理的配置项，参考[imageInputConfig.json](https://github.com/alibaba/MNN/blob/master/tools/quantization/imageInputConfig.json)，该Json的配置信息如下表所示：

|  key   |  value  |  说明  |
|--------|---------|-------|
| format |  "RGB", "BGR", "RGBA", "GRAY" | 图片统一按RGBA读取，然后转换到`format`指定格式 |
| mean/normal | `[float]` | `dst = (src - mean) * normal` |
| width/height | `int` | 模型输入的宽高 |
| path | `str` | 存放校正特征量化系数的图片目录 |
| used_image_num | `int` | 用于指定使用上述目录下多少张图片进行校正，默认使用`path`下全部图片 |
| feature_quantize_method | "KL", "ADMM", "EMA" | 指定计算特征量化系数的方法，默认："KL" |
| weight_quantize_method | "MAX_ABS", "ADMM" | 指定权值量化方法，默认："MAX_ABS" |
| feature_clamp_value | `int` | 特征的量化范围，默认为127，即[-127, 127]对称量化，有时，量化之后溢出会很多，造成误差较大，可适当减小此范围，如减小至120，但范围减小太多会导致分辨率下降，使用时需测试 |
| weight_clamp_value | `int` | 权值的量化范围，默认127，作用同feature_clamp_value，由于权值精度模型效果影响较大，建议调整feature_clamp_value即可 |
| batch_size | `int` | EMA方法中指定batch size，和模型训练时差不多 |
| quant_bits | `int` | 量化后的bit数，默认为8 |
| skip_quant_op_names | `[str]` | 跳过不量化的op的卷积op名字，因为有些层，如第一层卷积层，对模型精度影响较大，可以选择跳过不量化，可用netron可视化模型，找到相关op名字 |
| input_type | `str` | 输入数据的类型，默认为"image" |
| debug | `bool` | 是否输出debug信息，true或者false，输出的debug信息包含原始模型和量化模型各层输入输出的余弦距离和溢出率 |

| feature_quantize_method | 说明 |
|--------------------|------|
| KL | 使用KL散度进行特征量化系数的校正，一般需要100 ~ 1000张图片(若发现精度损失严重，可以适当增减样本数量，特别是检测/对齐等回归任务模型，样本建议适当减少) |
| ADMM | 使用ADMM（Alternating Direction Method of Multipliers）方法进行特征量化系数的校正，一般需要一个batch的数据 |
| EMA | 使用指数滑动平均来计算特征量化参数，这个方法会对特征进行非对称量化，精度可能比上面两种更好。这个方法也是[MNNPythonOfflineQuant](https://github.com/alibaba/MNN/tree/master/tools/MNNPythonOfflineQuant)的底层方法，建议使用这个方法量化时，保留你pb或onnx模型中的BatchNorm，并使用 --forTraining 将你的模型转到MNN，然后基于此带BatchNorm的模型使用EMA方法量化。另外，使用这个方法时batch size应设置为和训练时差不多最好。 |

| weight_quantize_method | 说明 |
|--------------------|------|
| MAX_ABS | 使用权值的绝对值的最大值进行对称量化 |
| ADMM | 使用ADMM方法进行权值量化 |

## 量化模型的使用
和浮点模型同样使用方法，输入输出仍然为浮点类型
## 参考资料
[Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16767/16728)
## 用法示例
```bash
cd /path/to/MNN/build
cmake -DMNN_BUILD_QUANTOOLS=ON && make -j4
./quantized.out mobilnet.mnn mobilnet_quant.mnn mobilnet_quant.json                                         
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/quantized.cpp:23: >>> modelFile: mobilnet.mnn
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/quantized.cpp:24: >>> preTreatConfig: mobilnet_quant.json
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/quantized.cpp:25: >>> dstFile: mobilnet_quant.mnn
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/quantized.cpp:53: Calibrate the feature and quantize model...
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/calibration.cpp:156: Use feature quantization method: KL
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/calibration.cpp:157: Use weight quantization method: MAX_ABS
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/calibration.cpp:177: feature_clamp_value: 127
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/calibration.cpp:178: weight_clamp_value: 127
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/Helper.cpp:111: used image num: 2
[11:53:29] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/calibration.cpp:668: fake quant weights done.
ComputeFeatureRange: 100.00 %
CollectFeatureDistribution: 100.00 %
[11:53:31] /Users/wangzhaode/copy/AliNNPrivate/tools/quantization/quantized.cpp:58: Quantize model done!
```
配置文件`mobilnet_quant.json`内容如下：
```json
{
    "format":"RGB",
    "mean":[
        103.94,
        116.78,
        123.68
    ],
    "normal":[
        0.017,
        0.017,
        0.017
    ],
    "width":224,
    "height":224,
    "path":"../resource/images/",
    "used_image_num":2,
    "feature_quantize_method":"KL",
    "weight_quantize_method":"MAX_ABS",
    "model":"mobilenet.mnn"
}
```
## Python版
我们提供了预编译的Quant Python工具：[mnnquant](python.html#mnnquant)
