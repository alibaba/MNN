# Python工具
Pymnn是MNN的Python版本，其中将部分MNN工具封装成了`MNNTools`，MNNTools模块主要有以下工具：
- mnn
- mnnconvert
- mnnquant

### mnn
mnn工具的功能是列出目前MNNTools支持的所有工具，使用如下：
```bash
mnn
mnn toolsets has following command line tools
    $mnn
        list out mnn commands
    $mnnconvert
        convert other model to mnn model
    $mnnquant
        quantize  mnn model
```
### mnnconvert
mnnconvert是对[MNNConvert](convert.md)的Python封装，参数使用可以[参考](convert.html#id2)，示例如下：
```bash
mnnconvert -f ONNX --modelFile mobilenetv2-7.onnx --MNNModel mobilenetv2-7.mnn --bizCode mobilenet
Start to Convert Other Model Format To MNN Model...
[11:34:53] :40: ONNX Model ir version: 6
Start to Optimize the MNN Net...
107 op name is empty or dup, set to Const107
108 op name is empty or dup, set to BinaryOp108
109 op name is empty or dup, set to Unsqueeze109
111 op name is empty or dup, set to Unsqueeze111
inputTensors : [ input, ]
outputTensors: [ output, ]
Converted Success!
```
### mnnquant
mnnquant是对[quantized.out](quant.html#id4)的Python封装，具体用法可以参考[quantized.out](quant.html#id4)，示例如下：
```bash
cp /path/to/MNN # using MNN/resource/images as input
mnnquant shuffle.mnn shuffle_quant.mnn shuffle_quant.json  
[11:48:17] :48: >>> modelFile: shuffle.mnn
[11:48:17] :49: >>> preTreatConfig: shuffle_quant.json
[11:48:17] :50: >>> dstFile: shuffle_quant.mnn
[11:48:17] :77: Calibrate the feature and quantize model...
[11:48:17] :156: Use feature quantization method: KL
[11:48:17] :157: Use weight quantization method: MAX_ABS
[11:48:17] :177: feature_clamp_value: 127
[11:48:17] :178: weight_clamp_value: 127
[11:48:17] :111: used image num: 2
[11:48:17] :668: fake quant weights done.
ComputeFeatureRange: 100.00 %
CollectFeatureDistribution: 100.00 %
[11:48:36] :82: Quantize model done!
```
配置文件`shuffle_quant.json`内容如下：
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
    "path":"./resource/images/",
    "used_image_num":2,
    "feature_quantize_method":"KL",
    "weight_quantize_method":"MAX_ABS",
    "model":"shuffle.mnn"
}
```