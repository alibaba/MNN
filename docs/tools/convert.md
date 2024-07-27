# 模型转换工具
[从源码编译](../compile/tools.html#id2)
## 参数说明
```bash
Usage:
  MNNConvert [OPTION...]

  -h, --help                    Convert Other Model Format To MNN Model

  -v, --version                 显示当前转换器版本
  
  -f, --framework arg           需要进行转换的模型类型, ex: [TF,CAFFE,ONNX,TFLITE,MNN,TORCH, JSON]
  
      --modelFile arg           需要进行转换的模型文件名, ex: *.pb,*caffemodel

      --batch arg               如果模型时输入的batch是动态的，可以指定转换后的batch数

      --keepInputFormat         是否保持原始模型的输入格式，默认为：是；

      --optimizeLevel arg       图优化级别，默认为1：
                                    - 0： 不执行图优化，仅针对原始模型是MNN的情况；
                                    - 1： 保证优化后针对任何输入正确；
                                    - 2： 保证优化后对于常见输入正确，部分输入可能出错；

      --optimizePrefer arg      图优化选项，默认为0：
                                    - 0：正常优化
                                    - 1：优化后模型尽可能小；
                                    - 2：优化后模型尽可能快；
      
      --prototxt arg            caffe模型结构描述文件, ex: *.prototxt
      
      --MNNModel arg            转换之后保存的MNN模型文件名, ex: *.mnn
      
      --fp16                    将conv/matmul/LSTM的float32参数保存为float16，
      													模型将减小一半，精度基本无损
      
      --bizCode arg             MNN模型Flag, ex: MNN
      
      --debug                   使用debug模型显示更多转换信息
      
      --forTraining             保存训练相关算子，如BN/Dropout，default: false
      
      --weightQuantBits arg     arg=2~8，此功能仅对conv/matmul/LSTM的float32权值进行量化，
      													仅优化模型大小，加载模型后会解码为float32，量化位宽可选2~8，
                                运行速度和float32模型一致。8bit时精度基本无损，模型大小减小4倍
                                default: 0，即不进行权值量化

      --weightQuantAsymmetric   与weightQuantBits结合使用，决定是否用非对称量化，默认为`true`
      
      --compressionParamsFile arg
                                使用MNN模型压缩工具箱生成的模型压缩信息文件或根据用户提供的量化参数来生成对应的量化模型，量化参数文件可参考tools/converter/user_provide_quant_params.json
                                
      --saveStaticModel         固定输入形状，保存静态模型， default: false

      --targetVersion arg       兼容旧的推理引擎版本，例如：1.2f

      --customOpLibs arg        用户自定义Op库，用于TorchScript模型中自定义算子的实现，如：libmy_add.so

      --info                    当-f MNN时，打印模型基本信息（输入名、输入形状、输出名、模型版本等）

      --authCode arg            认证信息，指定模型的认证信息，可用于鉴权等逻辑

      --inputConfigFile arg     保存静态模型所需要的配置文件, ex: ~/config.txt。文件格式为：
                                input_names = input0,input1
                                input_dims = 1x3x224x224,1x3x64x64

      --testdir arg             测试转换 MNN 之后，MNN推理结果是否与原始模型一致。
                                arg 为测试数据的文件夹，生成方式参考 "正确性校验" 一节

      --thredhold arg           当启用 --testdir 后，设置正确性校验的误差允可范围
                                若不设置，默认是 0.01

      --JsonFile arg            当-f MNN并指定JsonFile时，可以将MNN模型转换为Json文件

      --alignDenormalizedValue arg
                                可选值：{0, 1}， 默认为1, 当`float(|x| < 1.18e-38)`会被视为0

      --detectSparseSpeedUp arg
                                可选值：{0, 1}， 默认为1, 会检测权重是否使用稀疏化加速

      --saveExternalData        将权重，常量等数据存储在额外文件中，默认为`false`

```

**说明1: 选项weightQuantBits，使用方式为 --weightQuantBits numBits，numBits可选2~8，此功能仅对conv/matmul/LSTM的float32权值进行量化，仅优化模型大小，加载模型后会解码为float32，量化位宽可选2~8，运行速度和float32模型一致。经内部测试8bit时精度基本无损，模型大小减小4倍。default: 0，即不进行权值量化。**

**说明2：如果使用Interpreter-Session C++接口开发，因为NC4HW4便于与ImageProcess结合，可以考虑在转换模型时使用自动内存布局：`--keepInputFormat=0`**

## 其他模型转换到MNN
### TensorFlow to MNN
```bash
./MNNConvert -f TF --modelFile XXX.pb --MNNModel XXX.mnn --bizCode biz
```
注意：`*.pb`必须是frozen model，不能使用saved_model

### TensorFlow Lite to MNN
```bash
./MNNConvert -f TFLITE --modelFile XXX.tflite --MNNModel XXX.mnn --bizCode biz
```

### Caffe to MNN
```bash
./MNNConvert -f CAFFE --modelFile XXX.caffemodel --prototxt XXX.prototxt --MNNModel XXX.mnn --bizCode biz
```

### ONNX to MNN
```bash
./MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz
```
### TorchScript to MNN
```shell
./MNNConvert -f TORCH --modelFile XXX.pt --MNNModel XXX.mnn --bizCode biz
```
注意：TorchScript模型要求使用torch.jit导出的模型，**不要直接使用Pytorch的权重文件作为模型转换**；导出模型的代码如下：
```python
import torch
# ...
#  model is exported model
model.eval()
# trace
model_trace = torch.jit.trace(model, torch.rand(1, 3, 1200, 1200))
model_trace.save('model_trace.pt')
# script
model_script = torch.jit.script(model)
model_script.save('model_script.pt')
```

### MNN to Json
想了解MNN模型的具体结构，输入输出信息时，可以将模型转换为Json文件，并查找相关信息获取。
```bash
./MNNConvert -f MNN --modelFile XXX.mnn --JsonFile XXX.json
```
### Json to MNN
可以通过将MNN模型转换为Json文件，对Json文件进行编辑修改，然后在转换为MNN模型，达到对模型修改微调的目的。
```bash
./MNNConvert -f JSON --modelFile XXX.json --MNNModel XXX.mnn
```

## 正确性校验
为了便于开发者排查问题，对于 PB / Tflite / Onnx ，MNN 提供了正确性校验工具（位于 tools/scripts 目录），以检查 MNN 推理结果是否与 原始模型一致。
相关脚本为：

- testMNNFromTf.py ：适用 pb
- testMNNFromTflite.py ：适用 tflite
- testMNNFromOnnx.py ：适用 onnx
- testMNNFromTorch.py ：适用 pt (torchscript)

注意：

- 如果模型是动态输入形状，MNN 在脚本中默认不固定部分为1，有可能在 Tensorflow / OnnxRuntime / Torch 验证阶段报错。此时需要修改脚本中对应的输入部分，比如 testMNNFromOnnx.py 中的 run_onnx(self) 函数，把输入替换为有效的输入形状和内容。
- 对于由Torchscript转换的模型，一般都需要自行修改`testMNNFromTorch.py`中的的输入信息来测试。
- 如果模型输出层是 Identity 产生的，会因为 MNN 图优化的缘故丢失，此时需要校验上一层的输出，即在脚本后接输出名来测试，如: python3 ../tools/scripts/testMNNFromTf.py XXX.pb $NAME$


### 前置
- 测试 pb / tflite ：安装`tensorflow`(`pip install tensorflow`）
- 测试 onnx : 安装`onnxruntime`(`pip install onnxruntime`）
- 测试 torchscript：安装`torch`(`pip install torch`)
- 【可选】MNN模型转换工具编译完成（编译完成产生`MNNConvert`可执行文件）
### 使用
- 使用：在MNN的`build`目录下（包含`MNNConvert`）运行`python3 testMNNFromTf.py SRC.pb`（Onnx为`python3 testMNNFromOnnx.py SRC.onnx`，Tflite 类似），若最终结果为`TEST_SUCCESS`则表示 MNN 的模型转换与运行结果正确
- 若路径下面没有编译好的 MNNConvert 可执行文件，脚本会使用 pymnn 去进行校验
- 由于 MNN 图优化会去除 Identity ，有可能出现 find var error ，这个时候可以打开原始模型文件，找到 identity 之前的一层（假设为 LAYER_NAME ）校验，示例：`python3 ../tools/script/testMNNFromTF.py SRC.pb LAYER_NAME`；
- 完整实例如下（以onnx为例）：
  - 成功执行，当结果中显示`TEST_SUCCESS`时，就表示模型转换与推理没有错误
      ```bash
      cd build
      cmake -DMNN_BUILD_CONVERTER=ON .. && make -j4
      python ../tools/script/testMNNFromOnnx.py mobilenetv2-7.onnx # 模型转换后推理并与ONNXRuntime结果对比
      Dir exist
      onnx/test.onnx
      tensor(float)
      ['output']
      inputs:
      input
      onnx/
      outputs:
      onnx/output.txt (1, 1000)
      onnx/
      Test onnx
      Start to Convert Other Model Format To MNN Model...
      [21:09:40] /Users/wangzhaode/copy/AliNNPrivate/tools/converter/source/onnx/onnxConverter.cpp:40: ONNX Model ir version: 6
      Start to Optimize the MNN Net...
      108 op name is empty or dup, set to Const108
      109 op name is empty or dup, set to BinaryOp109
      110 op name is empty or dup, set to Unsqueeze110
      112 op name is empty or dup, set to Unsqueeze112
      97 op name is empty or dup, set to Unsqueeze97
      98 op name is empty or dup, set to Const98
      inputTensors : [ input, ]
      outputTensors: [ output, ]
      Converted Success!
      input
      output: output
      output: (1, 1000, )
      TEST_SUCCESS
      ```
- 默认只支持限定数值范围的输入随机生成，如需修改，请自己修改脚本
### 出错及解决
- 出现 Test Error 或者 MNN 的 crash 可直接反馈（提 github issue 或者钉钉群反馈）
- 如需自查，testMNNFromOnnx.py 提供 debug 功能，可方便定位出错的 layer / op ，示例：
   - python3 testMNNFromOnnx.py SRC.onnx DEBUG
- 示例，以ONNX为例：
   - 假设存在错误；此处为实验将MNN的Binary_ADD实现修改为错误实现；执行上述测试脚本，效果如下，显示`TESTERROR`表明可以转换但是推理结果有错误：
      ```bash
      python ../tools/script/testMNNFromOnnx.py mobilenetv2-7.onnx      
      Dir exist
      onnx/test.onnx
      tensor(float)
      ['output']
      inputs:
      input
      onnx/
      outputs:
      onnx/output.txt (1, 1000)
      onnx/
      Test onnx
      Start to Convert Other Model Format To MNN Model...
      [21:43:57] /Users/wangzhaode/copy/AliNNPrivate/tools/converter/source/onnx/onnxConverter.cpp:40: ONNX Model ir version: 6
      Start to Optimize the MNN Net...
      108 op name is empty or dup, set to Const108
      109 op name is empty or dup, set to BinaryOp109
      110 op name is empty or dup, set to Unsqueeze110
      112 op name is empty or dup, set to Unsqueeze112
      97 op name is empty or dup, set to Unsqueeze97
      98 op name is empty or dup, set to Const98
      inputTensors : [ input, ]
      outputTensors: [ output, ]
      Converted Success!
      input
      output: output
      output: (1, 1000, )
      TESTERROR output value error : absMaxV:5.814904 - DiffMax 32.684010
      Error for output output
      Save mnn result to  .error director
      ```
   - 对于推理出错的情况，可以使用`可视化工具`查看模型结果，测试每一层的输出，直至发现错误层：
      ```bash
      # test layer output 365: ERROR
      python ../tools/script/testMNNFromOnnx.py mobilenetv2-7.onnx 365
      ...
      365: (1, 32, 28, 28, )
      TESTERROR 365 value error : absMaxV:3.305553 - DiffMax 5.069034
      Error for output 365
      Save mnn result to  .error director
      # binary search test layers ...
      # test layer output 339: ERROR, 339's inputs is [489, 498]
      python ../tools/script/testMNNFromOnnx.py mobilenetv2-7.onnx 339 
      ...
      output: 339
      339: (1, 24, 56, 56, )
      TESTERROR 339 value error : absMaxV:3.704849 - DiffMax 5.504766
      Error for output 339
      Save mnn result to  .error director
      # test layer output 489: SUCCESS
      python ../tools/script/testMNNFromOnnx.py mobilenetv2-7.onnx 489
      ...
      output: 489
      489: (1, 24, 56, 56, )
      TEST_SUCCESS
      # test layer output 498: SUCCESS
      python ../tools/script/testMNNFromOnnx.py mobilenetv2-7.onnx 498
      ...
      output: 498
      498: (1, 24, 56, 56, )
      TEST_SUCCESS
      # so bug is layer 339
      ```
  - 对于ONNX的模型可以使用自动定位功能，在模型后输入`DEBUG`，便会执行基于支配树的二分查找，直至找到错误层：
      ```bash
      python ../tools/script/testMNNFromOnnx.py mobilenetv2-7.onnx DEBUG
      ...
      Test Node : Conv_14 True
      ### First Error Node is :  Add_15
      ```

## 算子支持列表
```bash
./MNNConvert -f CAFFE --OP
./MNNConvert -f TF --OP
./MNNConvert -f ONNX --OP
./MNNConvert -f TORCH --OP 
```

## 模型打印
将MNN模型文件dump成可读的类json格式文件，以方便对比原始模型参数，同时也可以对模型进行修改。
可以使用`MNNConvert`或`MNNDump2Json`将模型转换成Json文件；在对模型进行修改后还可以使用`MNNConvert`或`MNNRevert2Buffer`将Json文件转回MNN模型；执行方式如下：
```bash
./MNNDump2Json mobilenet_v1.mnn mobilenet_v1.json
# do some change in mobilenet_v1.json
./MNNRevert2Buffer mobilenet_v1.json mobilenet_v1_new.mnn
cat mobilenet_v1.json
{ "oplists":
[
{ "type": "Input", "name": "data", "outputIndexes":
[ 0 ]
, "main_type": "Input", "main":
{ "dims":
[ 1, 3, 224, 224 ]
, "dtype": "DT_FLOAT", "dformat": "NC4HW4" }
, "defaultDimentionFormat": "NHWC" }
,
{ "type": "Convolution", "name": "conv1", "inputIndexes":
[ 0 ]
, "outputIndexes":
[ 1 ]
, "main_type": "Convolution2D", "main":
{ "common":
{ "dilateX": 1, "dilateY": 1, "strideX": 2, "strideY": 2, "kernelX": 3, "kernelY": 3, "padX": 1, "padY": 1, "group": 1, "outputCount": 32, "relu": true, "padMode": "CAFFE", "relu6": false, "inputCount": 0 }
, weight: 
[ -0.0, -0.0, 0.0, -0.0, ... ]
, bias: 
[ -0.000004, 0.694553, 0.416608,  ... ]
 }
, "defaultDimentionFormat": "NHWC" }
, 
...
 ]
, "tensorName": 
[ "data", "conv1", "conv2_1/dw", "conv2_1/sep", "conv2_2/dw", "conv2_2/sep", "conv3_1/dw", "conv3_1/sep", "conv3_2/dw", "conv3_2/sep", "conv4_1/dw", "conv4_1/sep", "conv4_2/dw", "conv4_2/sep", "conv5_1/dw", "conv5_1/sep", "conv5_2/dw", "conv5_2/sep", "conv5_3/dw", "conv5_3/sep", "conv5_4/dw", "conv5_4/sep", "conv5_5/dw", "conv5_5/sep", "conv5_6/dw", "conv5_6/sep", "conv6/dw", "conv6/sep", "pool6", "fc7", "prob" ]
, "sourceType": "CAFFE", "bizCode": "AliNNTest", "tensorNumber": 0, "preferForwardType": "CPU" }
```

## Python版
我们提供了预编译的MNNConvert Python工具：[mnnconvert](python.html#mnnconvert)
