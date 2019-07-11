[English Version](Tools_EN.md)

# 工具说明

使用cmake编译时，build目录下的产物也包含测试使用的工具集，下面逐项说明。

## MNNV2Basic.out
### 功能
测试性能、输出结果，可检查与Caffe / Tensorflow的预期结果是否匹配。

### 参数
``` bash
./MNNV2Basic.out temp.mnn 10 0 0 1x3x224x224
```

- 第一个参数指定 待测试模型的二进制文件。
- 第二个参数指定 性能测试的循环次数，10就表示循环推理10次。
- 第三个参数指定 是否输出推理中间结果，0为不输出，1为输出；默认输出当前目录的output目录下。
- 第四个参数指定 执行推理的计算设备，有效值为 0（浮点 CPU）、1（Metal）、3（浮点OpenCL）、6（OpenGL），7(Vulkan)
- 第五个参数指定 输入tensor的大小，一般不需要指定。
- 其他

### 默认输入与输出
只支持单一输入、单一输出。输入为运行目录下的input_0.txt；输出为推理完成后的第一个输出tensor，转换为文本后，输出到output.txt中。


## checkFile.out
### 功能
检查两个tensor文本文件是否一致。

### 参数
``` bash
./checkFile.out XXX.txt YYY.txt 0.1
```

- 0.1 表示绝对阈值，不输入则为 0.0001 
- 比对值超过绝对阈值时，会直接输出到控制台


## checkDir.out
### 功能
比对两个文件夹下同名文件是否一致。

### 参数
``` bash
./checkDir.out output android_output 1
```

- 1 表示绝对阈值，不输入则为 0.0001 
- 比对值超过绝对阈值时，会直接输出到控制台


## timeProfile.out
### 功能
Op 总耗时统计工具和模型运算量估计。

### 参数
``` bash
./timeProfile.out temp.mnn 10 0 1x3x448x448
```

- 第一个参数 指定模型文件名
- 第二个参数 指定运行次数，默认 100
- 第三个参数 指定 执行推理的计算设备，有效值为 0（浮点 CPU）、1（Metal）、3（浮点OpenCL）、6（OpenGL），7(Vulkan)。（当执行推理的计算设备不为 CPU 时，Op 平均耗时和耗时占比可能不准）
- 第四个参数 指定输入大小，一般可不设

### 输出
- 第一列为 Op类型
- 第二列为 平均耗时
- 第三列为 耗时占比
- 示例: 
```
Node Type                Avg(ms)       %             Called times
Softmax                  0.018100      0.022775      1.000000
Pooling                  0.080800      0.101671      1.000000
ConvolutionDepthwise     14.968399     18.834826     13.000000
Convolution              64.404617     81.040726     15.000000
total time : 79.471924 ms, total mflops : 2271.889404
```

## backendTest.out
### 功能
这个工具可以对比指定计算设备和CPU执行推理的结果。

### 参数
``` bash
./backendTest.out temp.mnn 3 0.15
```

- 该工具默认读取当前目录下的 input_0.txt 作为输入
- 第一个参数：模型文件
- 第二个参数：执行推理的计算设备
- 第三个参数：误差容忍率
