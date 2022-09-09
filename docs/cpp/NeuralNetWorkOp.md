# NeuralNetWorkOp
```cpp
class NeuralNetWorkOp
```

## 成员函数

---
### _Input
```cpp
MNN_PUBLIC VARP _Input(INTS shape = {}, Dimensionformat data_format = NC4HW4, halide_type_t dtype = halide_type_of<float>());
```
创建一个输入变量

参数：
- `shape` 一个矢量，变量的形状
- `data_format` 一个枚举值，允许为NCHW/NHWC/NC4HW4
- `dtype` 目的变量的元素类型

返回：变量

---
### _Clone
```cpp
MNN_PUBLIC VARP _Clone(VARP source, bool deepCopy = false);
```
克隆变量


参数：
- `source` 被克隆的变量
- `deepCopy` 是否深度拷贝，默认为false

返回：与source类型相同的变量

---
### _Scalar
```cpp
MNN_PUBLIC VARP _Scalar(const void* ptr, halide_type_t type);
```
用于创建张量均值的标量类型


参数：
- `ptr` 一个指针，标量的值
- `type` 目标变量的数据类型

返回：标量类型

---
### _Const
```cpp
MNN_PUBLIC VARP _Const(float value, INTS shape = {}, Dimensionformat format = NHWC);
```
创建一个不可变变量

参数：
- `value` 显示的值
- `shape` 一个矢量，变量的形状
- `format` 一个枚举值，允许是NCHW/NHWC/NC4HW4

返回：一个不可变变量

---
### _Const
```cpp
MNN_PUBLIC VARP _Const(const void* ptr, INTS shape = {}, Dimensionformat format = NHWC,
                       halide_type_t type = halide_type_of<float>());
```
创建一个不可变变量

参数：
- `ptr` 一个指针，显示的值
- `shape` 一个矢量，变量的形状
- `format` 一个枚举值，允许是NCHW/NHWC/NC4HW4
- `type` 目标变量的数据类型

返回：一个不可变变量

---
### _TrainableParam
```cpp
MNN_PUBLIC VARP _TrainableParam(float value, INTS dims, Dimensionformat format);
```
可训练的参数


参数：
- `value` 参数的值
- `dims` 一个向量。置换后的索引轴顺序
- `format` 目标格式

返回：参数变量

---
### _TrainableParam
```cpp
MNN_PUBLIC VARP _TrainableParam(const void* ptr, INTS dims, Dimensionformat format,
                                halide_type_t type = halide_type_of<float>());
```
可训练的参数


参数：
- `ptr` 一个指针，参数的值
- `dims` 一个向量，置换后的索引轴顺序
- `format` 目标格式
- `type` 目标变量的数据类型

返回：参数变量

---
### _InnerProduct
```cpp
MNN_PUBLIC VARP _InnerProduct(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS outputShape);
```
全连接层，运算实质上是若干的输入向量与权值矩阵中的权值向量做內积的过程


参数：
- `weight` 权值矩阵
- `bias` 偏置项行向量
- `x` 输入变量
- `outputShape` 输出形状

返回：变量

---
### _Conv
```cpp
MNN_PUBLIC VARP _Conv(VARP weight, VARP bias, VARP x, PaddingMode pad = VALID, INTS stride = {1, 1},
                      INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});
```
对由多个输入平面组成的输入信号进行卷积


参数：
- `weight` 卷积产生的通道数
- `bias` 偏置项行向量，在输出中添加一个可学习的偏差
- `x` 输入变量
- `pad` 填充模式，默认为VALID
- `stride` 卷积步长，默认为{1, 1}
- `dilate` 扩张操作：控制kernel点（卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组卷积，默认不分组，为1组
- `pads` 填充操作，默认为{0, 0}

返回：卷积

---
### _Conv
```cpp
MNN_PUBLIC VARP _Conv(float weight, float bias, VARP x, INTS channel, INTS kernelSize, PaddingMode pad = VALID,
                      INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1);
```
对由多个输入平面组成的输入信号进行卷积


参数：
- `weight` 卷积产生的通道数
- `bias` 偏置项行向量，在输出中添加一个可学习的偏差
- `x` 输入变量
- `channel` 渠道
- `kernelSize` 卷积核大小
- `pad` 填充模式，默认为VALID
- `stride` 卷积步长，默认为{1, 1}
- `dilate` 扩张操作：控制kernel点（卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组卷积，默认不分组，为1组

返回：卷积

---
### _Conv
```cpp
MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0}, bool relu = false, bool relu6 = false, int nbits = 8);
```
对由多个输入平面组成的输入信号进行卷积


参数：
- `weight` 卷积产生的通道数
- `bias` 偏置项行向量，在输出中添加一个可学习的偏差
- `x` 输入变量
- `channel` 渠道
- `kernelSize` 卷积核大小
- `pad` 填充模式，默认为VALID
- `stride` 卷积步长，默认为{1, 1}
- `dilate` 扩张操作：控制kernel点（卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组卷积，默认不分组，为1组
- `pads` 填充操作，默认为{0, 0}
- `relu` 是否修正线性单元，默认为fasle
- `relu6` 修正线性单元6，默认为fasle
- `nbits` 默认为8

返回：卷积

---
### _Conv
```cpp
MNN_PUBLIC VARP _Conv(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0}, bool relu = false, bool relu6 = false);
```
对由多个输入平面组成的输入信号进行卷积


参数：
- `weight` 卷积产生的通道数
- `bias` 偏置项行向量，在输出中添加一个可学习的偏差
- `x` 输入变量
- `channel` 渠道
- `kernelSize` 卷积核大小
- `pad` 填充模式，默认为VALID
- `stride` 步长，默认为{1, 1}
- `dilate` 扩张操作：控制kernel点（卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组卷积，默认不分组，为1组
- `pads` 填充操作，默认为{0, 0}
- `relu` 是否修正线性单元，默认为fasle
- `relu6` 修正线性单元6，默认为fasle

返回：卷积

---
### _Deconv
```cpp
MNN_PUBLIC VARP _Deconv(VARP weight, VARP bias, VARP x, PaddingMode pad = VALID, INTS stride = {1, 1},
                                INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});
```
卷积的反向操作


参数：
- `weight` 反卷积产生的通道数
- `bias` 偏置项行向量，在输出中添加一个可学习的偏差
- `x` 输入变量
- `pad` 填充模式，默认为VALID
- `stride` 步长，默认为{1, 1}
- `dilate` 扩张操作：控制kernel点（反卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组反卷积，默认不分组，为1组
- `pads` 填充操作，默认为{0, 0}

返回：转置卷积（反卷积）

---
### _Deconv
```cpp
MNN_PUBLIC VARP _Deconv(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
PaddingMode pad, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0}, bool relu = false, bool relu6 = false);
```
卷积的反向操作


参数：
- `weight` 反卷积产生的通道数
- `bias` 偏置项行向量，在输出中添加一个可学习的偏差
- `x` 输入变量
- `channel` 渠道
- `kernelSize` 反积核大小
- `pad` 填充模式，默认为VALID
- `stride` 步长，默认为{1, 1}
- `dilate` 扩张操作：控制kernel点（反卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组反卷积，默认不分组，为1组
- `pads` 填充操作，默认为{0, 0}
- `relu` 是否修正线性单元，默认为fasle
- `relu6` 是否修正线性单元6，默认为fasle

返回：转置卷积（反卷积）

---
### _MaxPool
```cpp
MNN_PUBLIC VARP _MaxPool(VARP x, INTS kernel, INTS stride = {1, 1}, PaddingMode pad = VALID, INTS pads= {0, 0});
```
最大值池化操作


参数：
- `x` 池化的输入
- `kernel` 内核
- `stride` 窗口在每一个维度上滑动的步长
- `pad` 填充模式，默认为VALID
- `pads` 填充操作，默认为{0, 0}

返回：最大池化值

---
### _AvePool
```cpp
MNN_PUBLIC VARP _AvePool(VARP x, INTS kernel, INTS stride = {1, 1}, PaddingMode pad = VALID, INTS pads= {0, 0});
```
平均池化操作


参数：
- `x` 池化的输入
- `kernel` 内核
- `stride` 窗口在每一个维度上滑动的步长
- `pad` 填充模式，默认为VALID
- `pads` 填充操作，默认为{0, 0}

返回：平均池化值

---
### _Reshape
```cpp
MNN_PUBLIC VARP _Reshape(VARP x, INTS shape, Dimensionformat original_format = NCHW);
```
重塑一个变量

参数：
- `x` 被重塑的变量
- `shape` 一个矢量，目标的形状变量
- `original_format` 一个枚举值，只允许NCHW/NHWC，不允许NC4HW4，因为它提供额外的信息(x来自NCHW或NHWC)当x是NC4HW4时

返回：与' x '类型相同的变量

---
### _Reshape
```cpp
MNN_PUBLIC VARP _Reshape(VARP x, VARP shape);;
```
重塑一个变量

参数：
- `x` 被重塑的变量
- `shape` 一个矢量，目标的形状变量

返回：与' x '类型相同的变量

---
### _Scale
```cpp
MNN_PUBLIC VARP _Scale(VARP x, int channels, std::vector<float>&& scales, std::vector<float>&& bias);
```
返回`x * scale + bias`的值


参数：
- `x` 输入变量
- `channels` 渠道
- `scales` 输入变量
- `bias` 输入变量

返回：`x * scale + bias`的值

---
### _Relu
```cpp
MNN_PUBLIC VARP _Relu(VARP x, float slope = 0.0f);
```
给定一个输入值x，如果x > 0，它计算输出为x，如果x <= 0，则返回斜率 * x


参数：
- `x` 一个输入变量
- `slope` 一个浮点数，一个正的浮点值，它通过乘以“斜率”而不是设置为0.0f来漏掉负的部分，默认为0.0f

返回：与' x '类型相同的变量

---
### _Relu6
```cpp
MNN_PUBLIC VARP _Relu6(VARP x, float minValue = 0.0f, float maxValue = 6.0f);
```
给定一个输入值x，它计算修正线性6: min(max(x, 0)， 6)


参数：
- `x` 一个输入变量
- `minValue` 最小值
- `maxValue` 最大值

返回：与' x '类型相同的变量

---
### _PRelu
```cpp
MNN_PUBLIC VARP _PRelu(VARP x, std::vector<float> &&slopes);
```
给定一个输入值x，如果x > 0，它计算输出为x，如果x <= 0，则返回斜率 * x


参数：
- `x` 一个变量，必须是4-D的NC4HW4格式
- `slopes` 一个向量，保存大小为x

返回：与' x '类型相同的变量

---
### _Softmax
```cpp
MNN_PUBLIC VARP _Softmax(VARP logits, int axis = -1);
```
归一化指数函数，作用是将多分类的结果以概率的形式展现出来


参数：
- `logits` 一个非空的变量，必须Halide_Type_Float
- `axis` 默认值是-1，表示最后一个维度

返回：与' x '类型相同的变量

---
### _Softplus
```cpp
MNN_PUBLIC VARP _Softplus(VARP features);
```
激活函数，可以看作是ReLU函数的平滑：log(exp(features) + 1)

参数：
- `features` 一个变量，必须Halide_Type_Float

返回：与'features'类型相同的变量

---
### _Softsign
```cpp
MNN_PUBLIC VARP _Softsign(VARP features);
```
激活函数，是Tanh函数的另一个替代选择：features / (abs(features) + 1)


参数：
- `features` 一个变量，必须Halide_Type_Float

返回：与'features'类型相同的变量

---
### _Split
```cpp
MNN_PUBLIC std::vector<VARP> _Split(VARP value, INTS size_splits, int axis = 0);
```
将变量值拆分为子变量列表


参数：
- `value` 要拆分的变量
- `size_splits` 一个矢量，一个一维整数，包含每个输出变量沿轴的大小
- `axis` 一个int型，沿其进行分割的维度，必须在范围[-rank(value)， rank(value))内，默认值为0

返回：变量列表

---
### _Slice
```cpp
MNN_PUBLIC VARP _Slice(VARP x, VARP starts, VARP sizes);
```
返回从张量中提取想要的切片，此操作从由starts指定位置开始的张量x中提取一个尺寸sizes的切片，切片sizes被表示为张量形状，提供公式`x[starts[0]:starts[0]+sizes[0], ..., starts[-1]:starts[-1]+sizes[-1]]`


参数：
- `x` 输入变量
- `starts` 切片提取起始位置
- `sizes` 切片提取的尺寸

返回：切片数据

---
### _StridedSlice
```cpp
MNN_PUBLIC VARP _StridedSlice(VARP input, VARP begin, VARP end, VARP strided,
                              int32_t beginMask, int32_t endMask, int32_t ellipsisMask,
                              int32_t newAxisMask, int32_t shrinkAxisMask);
```
从给定的 x 张量中提取一个尺寸 (end-begin)/stride 的片段，从 begin 片段指定的位置开始，以步长 stride 添加索引，直到所有维度都不小于 end，这里的 stride 可以是负值，表示反向切片,公式：`x[begin[0]:strides[0]:end[0], ..., begin[-1]:strides[-1]:end[-1]]`。


参数：
- `input` 输入变量
- `begin` 开始切片处
- `end` 终止切片处
- `strided` 步长
- `beginMask` 输入变量，默认为0
- `endMask` 输入变量，默认为0
- `ellipsisMask` 输入变量，默认为0
- `newAxisMask` 输入变量，默认为0
- `shrinkAxisMask` 输入变量，默认为0

返回：提取的片段

---
### _StridedSliceWrite
```cpp
MNN_PUBLIC VARP _StridedSliceWrite(VARP input, VARP begin, VARP end, VARP strided, VARP write,
                                   int32_t beginMask, int32_t endMask, int32_t ellipsisMask,
                                   int32_t newAxisMask, int32_t shrinkAxisMask);
```
从给定的 x 张量中提取一个尺寸 (end-begin)/stride 的片段，从 begin 片段指定的位置开始，以步长 stride 添加索引，直到所有维度都不小于 end，这里的 stride 可以是负值，表示反向切片，可以写入。公式：`x[begin[0]:strides[0]:end[0], ..., begin[-1]:strides[-1]:end[-1]]`。


参数：
- `input` 输入变量
- `begin` 开始切片处
- `end` 终止切片处
- `strided` 步长
- `write` 输入变量
- `beginMask` 输入变量，默认为0
- `endMask` 输入变量，默认为0
- `ellipsisMask` 输入变量，默认为0
- `newAxisMask` 输入变量，默认为0
- `shrinkAxisMask` 输入变量，默认为0

返回：提取的片段

---
### _Concat
```cpp
MNN_PUBLIC VARP _Concat(VARPS values, int axis);
```
沿某个维度连接变量


参数：
- `values` 变量列表单个变量
- `axis` 一个int，要连接的维度，必须在范围[-rank(values)，rank(values))内。与在Python中一样，axis的索引是基于0的，区间[0,rank(values))为轴第1维，负轴表示轴+秩(值)-第维

返回：由输入变量连接而产生的变量

---
### _Convert
```cpp
MNN_PUBLIC VARP _Convert(VARP input, Dimensionformat format);
```
将变量转换为另一种格式(可能添加在' input '之后)


参数：
- `input` 输入变量
- `format` 目标格式

返回：一个变量，如果'input'已经是'format'，那么直接返回'input'，否则在'input'后面加上'format'变量

---
### _Transpose
```cpp
MNN_PUBLIC VARP _Transpose(VARP x, INTS perm);
```
转置x


参数：
- `x` 输入变量
- `perm` 一个向量，表示x的维数的排列

返回：转置变量

---
### _Transpose
```cpp
MNN_PUBLIC VARP _Transpose(VARP x, VARP perm);
```
转置x


参数：
- `x` 输入变量
- `perm` 一个向量，表示x的维数的排列

返回：转置变量

---
### _ChannelShuffle
```cpp
MNN_PUBLIC VARP _ChannelShuffle(VARP x, int group);
```
做以下操作：
    x = _Convert(x, NHWC);
    x = _Reshape(x, {0, 0, 0, group, -1}, NHWC);
    x = _Transpose(x, {0, 1, 2, 4, 3});
    x = _Reshape(x, {0, 0, 0, -1}, NHWC);
    channel_shuffle_res = _Convert(x, NC4HW4);


参数：
- `x` 输入变量
- `group` 控制分组

返回：一个变量，如果'input'已经是'format'，那么直接返回'input'，否则在'input'后面加上'format'变量

---
### _ChangeInputFormat
```cpp
MNN_PUBLIC VARP _ChangeInputFormat(VARP input, Dimensionformat format);
```
将变量转换为另一种格式(可能添加在' input '之前)


参数：
- `input` 输入变量
- `format` 目标格式

返回：目标变量，如果'input'已经是'format'，那么直接返回'input'，否则在'input'之前加上一个'format'变量。

---
### _Conv2DBackPropFilter
```cpp
MNN_PUBLIC VARP _Conv2DBackPropFilter(VARP input, VARP inputGrad, INTS kernelSize, PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});
```
计算卷积相对于滤波器的梯度


参数：
- `input` 4-D形状'[batch, in_height, in_width, in_channels]'
- `inputGrad` 输入梯度
- `kernelSize` 卷积核大小
- `pad` 填充模式，默认为VALID
- `stride` 滑动窗口的步长对每个维度的输入进行卷积，必须与格式指定的尺寸顺序相同
- `dilate` 扩张操作：控制kernel点（卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组卷积，默认不分组，为1组
- `pads` 填充操作，默认为{0, 0}

返回：梯度

---
### _PoolGrad
```cpp
MNN_PUBLIC VARP _PoolGrad(VARP originInput, VARP originOutput, VARP inputGrad, INTS kernel, INTS stride, PoolingMode type, PaddingMode pad = VALID, INTS pads= {0, 0});
```
池化操作


参数：
- `originInput` 起始输入
- `originOutput` 起始输出
- `inputGrad` 输入梯度
- `kernel` 内核
- `stride` 窗口在每一个维度上滑动的步长
- `type` 池化类型
- `pad` 填充模式，默认为VALID
- `pads` 填充操作，默认为{0, 0}

返回：池化的值

---
### _ReverseSequence
```cpp
MNN_PUBLIC VARP _ReverseSequence(VARP x, VARP y, int batchDim, int seqDim);
```
沿着batch_dim维度对x进行切片并反转维度seq_dim上的y[i]元素


参数：
- `x` 输入变量
- `y` 输入变量
- `batchDim` 切片的维度
- `seqDim` 反转的维度

返回：反转序列的值

---
### _Crop
```cpp
MNN_PUBLIC VARP _Crop(VARP images, VARP size, int axis, INTS offset);
```
裁剪图片


参数：
- `images` NC4HW4格式的4-D变量
- `size` 一个变量，它以'size'的形状作为输出裁剪变量的形状，而省略了'size'的值/格式
- `axis` 指示要裁剪的维度的整数，必须> = 2。所有在“axis”之前但不包括“axis”的维度都被保留，而包括和尾随“axis”的维度则被裁剪
- `offset` 表示偏移量的int型矢量，Length ('offset')必须为>=1且<=2。如果length('offset')为1，那么所有维度都将被这个量所抵消。否则，偏移量的数量必须等于每个维度中裁剪轴的数量

返回：NC4HW4格式的裁剪4-D变量

---
### _Resize
```cpp
MNN_PUBLIC VARP _Resize(VARP images, float xScale, float yScale);
```
调整图像


参数：
- `images` NC4HW4格式的4-D变量
- `xScale` 在x轴的缩放比例
- `yScale` 在y轴的缩放比例

返回：NC4HW4格式的调整大小的4-D变量

---
### _Pad
```cpp
MNN_PUBLIC VARP _Pad(VARP x, VARP paddings, PadValueMode mode = CONSTANT);
```
对张量进行填充


参数：
- `x` 输入变量
- `paddings` Halide_Type_Int类型的变量，形状为[n, 2]，其中n为变量的秩
- `mode` 一个枚举值，PadValueMode_CONSTANT、PadValueMode_SYMMETRIC或PadValueMode_REFLECT之一

返回：和x有相同的类型的变量

---
### _ExpandDims
```cpp
MNN_PUBLIC VARP _ExpandDims(VARP input, int axis);
```
返回在索引轴插入额外维度的变量


参数：
- `input` 输入变量
- `axis` 一个int，指定在其上展开输入形状的维度索引，给定一个D维的输入，轴必须在范围[-(D+1)， D](包括)

返回：具有与输入相同数据的变量，在轴指定的索引处插入额外的维度

---
### _ExpandDims
```cpp
MNN_PUBLIC VARP _ExpandDims(VARP input, VARP axis);
```
返回在索引轴插入额外维度的变量


参数：
- `input` 输入变量
- `axis` 一个int，指定在其上展开输入形状的维度索引，给定一个D维的输入，轴必须在范围[-(D+1)， D](包括)

返回：具有与输入相同数据的变量，在轴指定的索引处插入额外的维度

---
### _Shape
```cpp
MNN_PUBLIC VARP _Shape(VARP input, bool nchw = false);
```
返回变量的形状


参数：
- `input` 输入变量
- `nchw` 默认为false

返回：Halide_Type_Int类型的变量

---
### _Stack
```cpp
MNN_PUBLIC VARP _Stack(VARPS values, int axis=0);
```
将一列rank-R变量堆叠成一个rank-(R+1)变量，将'values'中的变量列表打包到一个比values中的每个变量都高1的变量中，通过沿轴尺寸排列它们，给定一个长度为N的形状变量列表(a, B, C)。
如果axis == 0，那么输出变量将具有形状(N, A, B, C)。
如果axis == 1，那么输出变量将具有形状(A, N, B, C)。


参数：
- `values` 具有相同形状和类型的变量对象列表
- `axis` 一个int，沿轴堆叠。默认为第一个维度。负值环绕，所以有效范围是[-(R+1)， R+1)

返回：一个与values相同类型的堆叠变量

---
### _CropAndResize
```cpp
MNN_PUBLIC VARP _CropAndResize(VARP image, VARP boxes, VARP box_ind, VARP crop_size, 
                                InterpolationMethod method, float extrapolation_value = 0.0);
```
从输入图像变量中提取作物，并使用双线性采样或最近邻采样(可能会改变长宽比)调整它们的大小，到由crop_size指定的通用输出大小。返回一个带有农作物的变量，该变量来自于框中边界框位置定义的输入图像位置。裁剪的盒子都被调整大小(双线性或最近邻插值)为固定大小= [crop_height, crop_width]。结果是一个4-D张量[num_boxes, crop_height, crop_width, depth](假设是NHWC格式)。


参数：
- `image` 一个形状[batch, image_height, image_width, depth]的4-D变量(假设NHWC格式)。image_height和image_width都必须为正
- `boxes` 形状为[num_boxes, 4]的二维变量。变量的第i行指定了box_ind[i]图像中盒子的坐标，并以规范化坐标[y1, x1, y2, x2]指定将y的一个归一化坐标值映射到y * (image_height - 1)处的图像坐标，因此，在图像高度坐标中，归一化图像高度的[0,1]区间映射为[0,image_height - 1]。我们允许y1 > y2，在这种情况下，采样的裁剪是原始图像的上下翻转版本。宽度维度的处理方式类似。允许在[0,1]范围之外的规范化坐标，在这种情况下，我们使用extrapolation_value来外推输入图像值。
- `box_ind` 形状为[num_boxes]的1-D变量，其int值范围为[0,batch)，box_ind[i]的值指定第i个框所指向的图像。
- `crop_size` 一个包含2个元素的1-D变量，size = [crop_height, crop_width]。所有裁剪的图像补丁都调整到这个大小。不保留图像内容的长宽比。作物高度和作物宽度都必须是正的。
- `method` 一个枚举值, CropAndResizeMethod_NEAREST或CropAndResizeMethod_BILINEAR，默认为CropAndResizeMethod_BILINEAR，extrapolation_value:适用时用于外推的值。
- `extrapolation_value` 推断值，默认为0.0

返回：形状[num_boxes, crop_height, crop_width, depth]的4-D变量(假设NHWC格式)

---
### _Fill
```cpp
MNN_PUBLIC VARP _Fill(VARP dims, VARP value);
```
创建一个填充标量值的变量


参数：
- `dims` 一个变量，必须是1-D Halide_Type_Int。表示输出变量的形状
- `value` 一个变量，0-D(标量)，值填充返回的变量

返回：一个变量，类型与值相同

---
### _Tile
```cpp
MNN_PUBLIC VARP _Tile(VARP input, VARP multiples);
```
通过平铺给定变量来构造一个变量


参数：
- `Fill` 一个变量，一维或更高
- `multiples` 一个变量，必须是1-D Halide_Type_Int，长度必须与输入的维度数相同

返回：一个变量，与输入的类型相同

---
### _Gather
```cpp
MNN_PUBLIC VARP _Gather(VARP params, VARP indices);
```
根据索引从参数中收集切片


参数：
- `params` 收集值的变量
- `indices` 指标变量，必须是范围[0,ndim (params)-1]的Halide_Type_Int

返回：从索引给出的索引中收集的参数值

---
### _GatherV2
```cpp
MNN_PUBLIC VARP _GatherV2(VARP params, VARP indices, VARP axis = nullptr);
```
根据索引从参数轴收集切片


参数：
- `params` 收集值的变量
- `indices` 指标变量，必须是范围[0,ndim (params)-1]的Halide_Type_Int
- `axis` 一个int，参数中用于收集下标的轴，支持负索引，如果设为0，它就和_Gather一样。目前只支持0

返回：从索引给出的索引收集的参数值

---
### _Squeeze
```cpp
MNN_PUBLIC VARP _Squeeze(VARP input, INTS axis = {});
```
从变量的形状中移除大小为1的维度


参数：
- `input` 一个变量，挤压输入
- `axis` 一个向量，默认为{}。如果指定，只挤压所列的尺寸。维度索引从0开始。必须在范围内[-rank(input)， rank(input))

返回：一个变量，与输入的类型相同。包含与输入相同的数据，但删除了一个或多个大小为1的维度

---
### _Unsqueeze
```cpp
MNN_PUBLIC VARP _Unsqueeze(VARP input, INTS axis = {});
```
插入到指定位置的尺寸为1的新数据


参数：
- `input` 输入变量
- `axis` 默认为{}，用来指定要增加的为1的维度

返回：变换后的数据

---
### _BatchToSpaceND
```cpp
MNN_PUBLIC VARP _BatchToSpaceND(VARP input, VARP block_shape, VARP crops);
```
BatchToSpace用于N-D变量，该操作将“batch”维度0重塑为形状block_shape + [batch]的M + 1个维度，将这些块插入由空间维度定义的网格中[1，…]M]，获得与输入相同秩的结果。这个中间结果的空间维度然后根据作物选择性地裁剪产生的输出。这与SpaceToBatch正好相反。请参阅下面的精确描述。


参数：
- `input` 必须为4-D格式，且为NC4HW4格式。N-D的形状input_shape = [batch] + spatial_shape + remaining_shape，其中spatial_shape有M个维度。
- `block_shape` 形状为[M]的1- d，所有值必须为>= 1
- `crops` 形状为[M, 2]的二维，所有值必须为>= 0。Crops [i] = [crop_start, crop_end]指定从输入维度i + 1开始的作物数量，对应空间维度i，要求crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]。
该操作相当于以下步骤:
shape的整形输入:[block_shape[0]，…]， block_shape[M-1]， batch / prod(block_shape)，input_shape[1],……, input_shape [n]]置换变形的尺寸产生置换的形状：
[batch / prod(block_shape)，input_shape[1]， block_shape[0]，…], input_shape [M], block_shape [M - 1], input_shape [M + 1],…, input_shape [n]]
重塑排列以产生形状的重塑：
[batch / prod(block_shape)，input_shape[1] * block_shape[0]，…], input_shape [M] * block_shape [M - 1], input_shape [M + 1],…, input_shape [n]]
裁剪开始和结束的维度[1，…， M]的reshaped_per组合根据作物产生形状的输出:
[batch / prod(block_shape)，input_shape[1] * block_shape[0] -庄稼[0,0]-庄稼[0,1]，…], input_shape [M] * block_shape [M - 1] -作物[M - 1,0]作物(M - 1, - 1), input_shape [M + 1],…, input_shape [n]]
例子:
对于以下形状[4,1,1,3]，block_shape = [2,2]， and crops =[[0,0]，[0,0]]的输入:
[[[[1, 2, 3]]],[[[4、5、6]]],[[[7 8 9]]],[[[10 11 12]]]]
输出变量的形状为[1,2,2,3]，值为:
X = [[[[1,2,3]， [4,5,6]，
[[7, 8, 9]， [10, 11, 12]]]]

返回：输出变量

---
### _GatherND
```cpp
MNN_PUBLIC VARP _GatherND(VARP params, VARP indices);
```
将参数中的切片收集到一个由索引指定形状的变量中


参数：
- `input` 一个变量，用于收集值的变量
- `indices` 一个变量，Halide_Type_Int类型

返回：一个变量，与参数有相同的类型

---
### _GatherElements
```cpp
MNN_PUBLIC VARP _GatherElements(VARP params, VARP indices);
```
返回将参数中的切片收集到一个由索引指定形状的变量


参数：
- `params` 输出参数
- `indices` 一个变量，Halide_Type_Int类型

返回：一个变量，与参数有相同的类型

---
### _GatherElements
```cpp
MNN_PUBLIC VARP _GatherElements(VARP params, VARP indices, VARP axis);
```
返回将参数中的切片收集到一个由索引指定形状的变量


参数：
- `params` 输出参数
- `indices` 一个变量，Halide_Type_Int类型
- `axis` 操作的维度

返回：一个变量，与参数有相同的类型

---
### _Selu
```cpp
MNN_PUBLIC VARP _Selu(VARP features, float scale, float alpha);
```
计算缩放指数线性:scale * alpha * (exp(特征)- 1)，如果< 0,scale * 特征


参数：
- `features` Halide_Type_Float类型的变量
- `scale` 比例因子(正浮点数)
- `alpha` 透明度因子(正浮动)

返回：一个变量，具有与功能相同的类型

---
### _Size
```cpp
MNN_PUBLIC VARP _Size(VARP input);
```
计算变量的大小


参数：
- `input` Halide_Type_Float或Halide_Type_Int类型的变量

返回：一个变量，形状是()，类型是Halide_Type_Int

---
### _Elu
```cpp
MNN_PUBLIC VARP _Elu(VARP features, float alpha=1.0);
```
计算指数线性:alpha * (exp(特征)- 1)，如果< 0，则特征为其他，Halide_Type_Float类型的变量


参数：
- `features` 一个变量，必须Halide_Type_Float
- `alpha` Alpha因子(正浮动)

返回：一个变量，具有与功能相同的类型

---
### _Threshold
```cpp
MNN_PUBLIC VARP _Threshold(VARP features, float threshold);
```
给定一个输入值x，如果x > threshold，它计算输出为1.0，如果x <= threshold，则为0.0


参数：
- `features` 一个变量，必须Halide_Type_Float
- `threshold` 阈值

返回：一个变量，具有与功能相同的类型

---
### _MatrixBandPart
```cpp
MNN_PUBLIC VARP _MatrixBandPart(VARP input, VARP num_lower, VARP num_upper);
```
复制一个变量，设置每个最内层矩阵中中心带以外的所有内容


参数：
- `input` 秩为k的变量
- `num_lower` 要保留的子对角线数。如果是负的，保持整个下三角形
- `num_upper` 要保持的超对角线的数量。如果是负的，保持整个上三角形

返回：将k变量的形状与输入相同，提取的带状张量

---
### _Moments
```cpp
MNN_PUBLIC std::vector<VARP> _Moments(VARP x, INTS axis, VARP shift, bool keepDims);
```
计算x的均值和方差


参数：
- `x` 一个变量，必须是4-D的NC4HW4格式
- `axis` 整数的数组。计算平均值和方差的轴。忽略此实现:必须为{2,3}
- `shift` 在当前实现中未使用
- `keepDims` 产生与输入相同维度的力矩。忽略此实现:必须为true

返回：均值和方差

---
### _SetDiff1D
```cpp
MNN_PUBLIC VARP _SetDiff1D(VARP x, VARP y);
```
计算两个数字或字符串列表之间的差值，给定一个列表x和一个列表y，该操作返回一个列表，该列表表示在x中但不在y中所有的值。返回的列表的排序顺序与x中数字出现的顺序相同(保留重复的数)。此操作还返回一个列表idx，该列表表示x中每个out元素的位置。


参数：
- `x` Halide_Type_Int类型的1-D变量
- `y` Halide_Type_Int类型的1-D变量，值删除

返回：Halide_Type_Int类型的1-D变量。值在x中存在，但在y中不存在

---
### _SpaceToDepth
```cpp
MNN_PUBLIC VARP _SpaceToDepth(VARP input, int block_size);
```
重新排列空间数据块，进入深度。更具体地说，它输出输入变量的副本，其中高度和宽度维度的值被移动到深度维度。block_size表示输入块的大小。大小为block_size x block_size的非重叠块在每个位置重新排列为深度。输出变量的深度是block_size * block_size * input_depth。每个输入块中的Y、X坐标成为输出通道索引的高阶分量。输入变量的高度和宽度必须能被block_size整除


参数：
- `input` 输入变量
- `block_size` 一个整数>= 2。空间块的大小

返回：一个变量。与输入的类型相同

---
### _SpaceToBatchND
```cpp
MNN_PUBLIC VARP _SpaceToBatchND(VARP input, VARP block_shape, VARP paddings);
```
他的操作划分了“空间”维度[1，…， M]输入到形状块block_shape的网格中，并将这些块与“批处理”维度交织使得在输出中，空间维度[1，…]， M]对应网格内的位置，批处理尺寸结合了空间块内的位置和原始批处理位置。在划分为块之前，输入的空间维度可以根据填充值选择零填充。请参阅下面的精确描述


参数：
- `input` 一个变量。必须为4-D格式，且为NC4HW4格式。N-D的形状input_shape = [batch] + spatial_shape + remaining_shape，其中spatial_shape有M个维度。block_shape:一个变量。必须是以下类型之一:int32, int64。形状为[M]的1- d，所有值必须为>= 1。
- `block_size` 一个整数>= 2。空间块的大小
- `paddings` 一个变量。必须是以下类型之一:int32, int64。形状为[M, 2]的二维，所有值必须为>= 0。padding [i] = [pad_start, pad_end]指定输入维度i + 1的填充，对应空间维度i。要求block_shape[i]除input_shape[i + 1] + pad_start + pad_end。

返回：一个变量。与输入的类型相同

---
### _ZerosLike
```cpp
MNN_PUBLIC VARP _ZerosLike(VARP input);
```
创建一个所有元素都设为零的变量


参数：
- `input` 输入变量

返回：一个所有元素都设为零的变量

---
### _Unstack
```cpp
MNN_PUBLIC std::vector<VARP> _Unstack(VARP value, int axis=0);
```
将秩为r的张量的给定维度解包为秩-(R-1)变量。
例如，给定一个形状变量(a, B, C, D)：
如果axis == 0，那么输出中的第i个变量是切片值[i，:，:，:]，输出中的每个变量将具有形状(B, C, D)(注意，与拆分不同，沿着拆分的维度消失了)。
如果axis == 1，那么输出中的第i个变量是切片值[:，i，:，:]，输出中的每个变量都有形状(A, C, D)。


参数：
- `value` 一个秩为R>0的变量
- `axis` 一个int。沿轴线解叠。默认为第一个维度。负值环绕，所以有效范围是[-R, R)

返回：从值中分离出来的变量对象列表

---
### _Rank
```cpp
MNN_PUBLIC VARP _Rank(VARP input);
```
返回变量的秩，返回一个0-D的int32变量，表示输入的秩。
注意:变量的秩与矩阵的秩是不同的。
它是唯一选择变量的每个元素所需的索引数。它也被称为“顺序”、“程度”或“ndim”。


参数：
- `input` 输入变量

返回：Halide_Type_Int类型的0-D变量

---
### _Range
```cpp
MNN_PUBLIC VARP _Range(VARP start, VARP limit, VARP delta);
```
创建一个数字序列


参数：
- `start` 0-D变量(标量)
- `limit` 0-D变量(标量)
- `delta` 0-D变量(标量)

返回：数字序列

---
### _DepthToSpace
```cpp
MNN_PUBLIC VARP _DepthToSpace(VARP input, int block_size);
```
将深度数据重新排列为空间数据块。这是SpaceToDepth的反向转换。更具体地说，它输出输入变量的副本，其中深度维度的值在空间块中移动到高度和宽度维度


参数：
- `input` 输入变量
- `block_size` 一个整数>= 2。空间块的大小，与Space2Depth相同

返回：一个变量。与输入的类型相同

---
### _PriorBox
```cpp
MNN_PUBLIC VARP _PriorBox(VARP feature, VARP image, 
                          std::vector<float> min_size, std::vector<float> max_size, std::vector<float>aspect_ratio, 
                          bool flip, bool clip, std::vector<float>variance,
                          unsigned int img_h, unsigned int img_w, float step_h, float step_w, float offset = 0.5);
```
SSD网络的priorbox层，人脸检测网络


参数：
- `feature` 一个变量。包含特性图，也就是caffprior中的底部[0]
- `image` 一个变量。包含图像，也就是caffe中的底部[1]
- `min_size` 最小区域大小(像素)
- `max_size` 最大区域大小(像素)
- `aspect_ratio` 各种纵横比。重复比率被忽略。如果没有提供，则使用默认1.0
- `flip` 如果为true，则翻转每个纵横比。例如，如果有高宽比“r”，也会生成高宽比“1.0/r”，违约事实
- `clip` 如果为true，则剪辑之前，使其在[0,1]内。默认为false。
- `variance` 在bboxes之前调整方差
- `img_h` 图像的高度。如果为0，则使用图像中的信息
- `img_w` 图像的宽度。如果为0，则使用图像中的信息
- `step_h` 步高
- `step_w` 步宽
- `offset` 每个单元格的左上角的偏移

返回：一个变量

---
### _Permute
```cpp
MNN_PUBLIC VARP _Permute(VARP input, INTS dims);
```
SSD网络的交换层，用于置换索引轴顺序的


参数：
- `input` 一个变量。包含特性图，也就是caffe中的底部[0]
- `dims` 一个向量。置换后的索引轴顺序

返回：一个变量

---
### _DetectionOutput
```cpp
MNN_PUBLIC VARP _DetectionOutput(VARP location, VARP confidence, VARP priorbox, 
                        unsigned int num_classes, bool share_location, int background_label_id, 
                        float nms_threshhold, int nms_topk, int code_type, 
                        bool variance_encoded_in_target,
                        int keep_top_k, float confidence_threshold, float visualize_threshold);
```
SSD网络的detectionoutput层，用于整合预选框、预选框偏移以及得分三项结果，最终输出满足条件的目标检测框、目标的label和得分


参数：
- `location` 位置
- `confidence` 得分
- `priorbox` SSD网络的priorbox层，人脸检测网络
- `num_classes` 预测种类
- `share_location` 指示不同类之间是否共享位置，默认为true
- `background_label_id` 默认为0
- `nms_threshhold` nms的阈值
- `nms_topk` nms的topk
- `code_type` 表示bbox编码模式，默认= CORNER
- `variance_encoded_in_target` variance是否被编码，默认为false
- `keep_top_k` 每张图片在nms处理后保留框的数量，默认值-1(保留所有boxes)
- `confidence_threshold` 得分阈值
- `visualize_threshold` 阈值用于将检测结果可视化

返回：目标检测框、目标的label和得分

---
### _DetectionPostProcess
```cpp
MNN_PUBLIC  std::vector<VARP> _DetectionPostProcess(VARP encode_boxes, VARP class_predictions, VARP anchors, 
                        int num_classes, int max_detections, 
                        int max_class_per_detection, int detections_per_class, 
                        float nms_threshold, float iou_threshold, 
                        bool use_regular_nms, std::vector<float> centersize_encoding);
```
SSD网络的detectionpostprocess层，对于预测阶段，模型输出的结果比较多，需要筛选最终结果


参数：
- `encode_boxes` 盒子编码
- `class_predictions` 类预测
- `anchors` 一个变量
- `num_classes` 预测种类
- `max_detections` 最大检测次数
- `max_class_per_detection` 每次检测的最大类
- `detections_per_class` 表示每个类的检测
- `nms_threshold` nms的阈值
- `iou_threshold` iou的阈值
- `use_regular_nms` 是否使用常规的NMS方法，目前只支持false
- `centersize_encoding` 一个浮点向量，表示中心大小编码

返回：4个变量，detection_boxes, detection_class, detection_scores, num_detections

---
### _Interp
```cpp
MNN_PUBLIC VARP _Interp(VARPS xs, float widthScale, float heightScale, int outputWidth, int outputHeight, int resizeType, bool alignCorners);
```
一维线性插值，返回离散数据的一维分段线性插值结果


参数：
- `xs` 待插入数据的横坐标
- `widthScale` 宽度比
- `heightScale` 高度比
- `outputWidth` 输出宽度
- `outputHeight` 输出高度
- `resizeType` 调整类型
- `alignCorners` 是否边缘对齐

返回：离散数据的一维分段线性插值结果

---
### _ZeroGrad
```cpp
MNN_PUBLIC VARP _ZeroGrad(VARP x);
```
参数梯度置0.


参数：
- `x` 输入变量

返回：梯度为0的变量

---
### _Conv
```cpp
MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<int>&& bias, std::vector<float>&& scale, VARP x, INTS channel, INTS kernelSize, PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu, int nbits = 8);
```
对由多个输入平面组成的输入信号进行卷积


参数：
- `weight` 卷积产生的通道数
- `bias` 偏置项行向量，在输出中添加一个可学习的偏差
- `scale` 缩放因子
- `x` 输入变量
- `channel` 渠道
- `kernelSize` 卷积核大小
- `pad` 填充模式，默认为VALID
- `stride` 卷积步长，默认为{1, 1}
- `dilate` 扩张操作：控制kernel点（卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组卷积，默认不分组，为1组
- `pads` 填充操作，默认为{0, 0}
- `relu` 是否修正线性单元，默认为fasle
- `nbits`  默认为8

返回：卷积

---
### _Conv
```cpp
MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<int>&& bias, std::vector<float>&& scale,
                      VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu,
                      int8_t inputZeroPoint, int8_t outputZeroPoint,
                      int8_t minValue, int8_t maxValue, bool accumulateToInt16);
```
对由多个输入平面组成的输入信号进行卷积


参数：
- `weight` 卷积产生的通道数
- `bias` 偏置项行向量，在输出中添加一个可学习的偏差
- `scale` 缩放因子
- `x` 输入变量
- `channel` 渠道
- `kernelSize` 卷积核大小
- `pad` 填充模式，默认为VALID
- `stride` 卷积步长，默认为{1, 1}
- `dilate` 扩张操作：控制kernel点（卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组卷积，默认不分组，为1组
- `pads` 填充操作，默认为{0, 0}
- `relu` 是否修正线性单元，默认为fasle
- `inputZeroPoint` 输入变量
- `outputZeroPoint` 输入变量
- `minValue` 最小值
- `maxValue` 最大值
- `accumulateToInt16` 输入变量

返回：卷积

---
### _Conv
```cpp
MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<float>&& bias, std::vector<float>&& weightScale,
                      VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu,
                      float scaleIn, float scaleOut,
                      int8_t inputZeroPoint, int8_t outputZeroPoint,
                      int8_t minValue, int8_t maxValue, float weightClampValue, bool accumulateToInt16);
```
对由多个输入平面组成的输入信号进行卷积


参数：
- `weight` 卷积产生的通道数
- `bias` 偏置项行向量，在输出中添加一个可学习的偏差
- `weightScale` 卷积产生的通道数的缩放因子
- `x` 输入变量
- `channel` 渠道
- `kernelSize` 卷积核大小
- `pad` 填充模式，默认为VALID
- `stride` 卷积步长，默认为{1, 1}
- `dilate` 扩张操作：控制kernel点（卷积核点）的间距，默认值为{1, 1}
- `group` 控制分组卷积，默认不分组，为1组
- `pads` 填充操作，默认为{0, 0}
- `relu` 是否修正线性单元，默认为fasle
- `scaleIn` 向内扩展值
- `scaleOut` 向外扩展值
- `inputZeroPoint` 输入变量
- `outputZeroPoint` 输入变量
- `minValue` 最小值
- `maxValue` 最大值
- `weightClampValue` 输入变量
- `accumulateToInt16` 输入变量

返回：卷积

---
### _CosineSimilarity
```cpp
MNN_PUBLIC VARP _CosineSimilarity(VARP input0, VARP input1, VARP inputDim);
```
余弦相似度，又称为余弦相似性，是通过测量两个向量的夹角的余弦值来度量它们之间的相似性


参数：
- `input0` 输入变量
- `input1` 输入变量
- `inputDim`表示对应行或者列的向量之间进行cos相似度计算

返回：和input0类型一致的数据

---
### _GridSample
```cpp
MNN_PUBLIC VARP _GridSample(VARP input, VARP grid, InterpolationMethod mode=BILINEAR, GridSamplePaddingMode paddingMode=GRID_SAMPLE_PADDING_ZEROS, bool alignCorners=false);
```
提供一个input的Tensor以及一个对应的flow-field网格(比如光流，体素流等)，然后根据grid中每个位置提供的坐标信息(这里指input中pixel的坐标)，将input中对应位置的像素值填充到grid指定的位置，得到最终的输出。


参数：
- `input` 输入数据
- `grid` flow-field网格
- `mode` 定义了在input中指定位置的pixel value中进行插值的方法，默认为BILINEAR
- `paddingMode` 对于越界的位置在网格中采用填充方式，默认为GRID_SAMPLE_PADDING_ZEROS
- `alignCorners` 默认为false

返回：网格数据

---
### _FloatToInt8
```cpp
MNN_PUBLIC VARP _FloatToInt8(VARP x, VARP scale, char minValue, char maxValue);
```
float类型转换为Int8类型


参数：
- `x` 输入变量
- `scale` 比例因子
- `minValue` 最小值
- `maxValue` 最大值

返回：Int8类型数据

---
### _FloatToInt8
```cpp
MNN_PUBLIC VARP _FloatToInt8(VARP x, VARP scale, int8_t minValue, int8_t maxValue, int8_t zeroPoint);
```
float类型转换为Int8类型


参数：
- `x` 输入变量
- `scale` 比例因子
- `minValue` 最小值
- `maxValue` 最大值
- `zeroPoint` 原点

返回：Int8类型数据

---
### _Int8ToFloat
```cpp
MNN_PUBLIC VARP _Int8ToFloat(VARP x, VARP scale);
```
Int8转换为float类型


参数：
- `x` 输入变量
- `scale` 比例因子

返回：float类型数据

---
### _Int8ToFloat
```cpp
MNN_PUBLIC VARP _Int8ToFloat(VARP x, VARP scale, int8_t zeroPoint);
```
Int8转换为float类型


参数：
- `x` 输入变量
- `scale` 比例因子
- `zeroPoint` 原点

返回：float类型数据

---
### _Select
```cpp
MNN_PUBLIC VARP _Select(VARP select, VARP input0, VARP input1);
```
返回根据'cond'从'x'或'y'中选择的元素


参数：
- `select` 输入变量
- `input0` 输入变量
- `input1` 输入变量

返回：根据条件选中的元素

---
### _TopKV2
```cpp
MNN_PUBLIC std::vector<VARP> _TopKV2(VARP input0, VARP input1);
```
查找排序后的数据的值和索引。



参数：
- `input0` 输入变量
- `input1` 输入变量

返回：值和索引

---
### _ImageProcess
```cpp
MNN_PUBLIC VARP _ImageProcess(VARP input, CV::ImageProcess::Config config, CV::Matrix matrix, int oh, int ow, int oc, int dtype, uint8_t padVal = 0);
```
图像处理


参数：
- `input` 输入变量
- `config` 配置信息
- `matrix` 输出矩阵
- `oh` 图像高
- `ow` 图像宽
- `oc` 卷积计算类型
- `dtype` 返回数据类型
- `padVal` 默认为0

返回：图像

---
### _Where
```cpp
MNN_PUBLIC VARP _Where(VARP x);
```
返回满足条件`x > 0`的索引


参数：
- `x` 输入变量

返回：索引

---
### _Sort
```cpp
MNN_PUBLIC VARP _Sort(VARP x, int axis = -1, bool arg = false, bool descend = false);
```
排序


参数：
- `x` 输入变量
- `axis` 输入变量，int类型，操作的坐标轴，默认为-1
- `arg` 是否返回排序元素的index， 默认为false
- `descend` true代表倒序，false代表正序，默认为false

返回：排序结果

---
### _Raster
```cpp
MNN_PUBLIC VARP _Raster(const std::vector<VARP>& vars, const std::vector<int>& regions, const std::vector<int>& shape);
```
光栅化


参数：
- `vars` 输入变量
- `regions` 区域
- `shape` 输出形状

返回：光栅化的值

---
### _Nms
```cpp
MNN_PUBLIC VARP _Nms(VARP boxes, VARP scores, int maxDetections, float iouThreshold = -1, float scoreThreshold = -1);
```
非极大值抑制算法，搜索局部极大值，抑制非极大值元素


参数：
- `boxes` 形状必须为[num, 4]
- `scores` float类型的大小为[num_boxes]代表上面boxes的每一行，对应的每一个box的一个score
- `maxDetections` 一个整数张量，代表最多可以利用NMS选中多少个边框
- `iouThreshold` IOU阙值展示的是否与选中的那个边框具有较大的重叠度，默认为-1
- `scoreThreshold` 默认为-1，来决定什么时候删除这个边框

返回：搜索局部极大值，抑制非极大值元素

---
### _Im2Col
```cpp
MNN_PUBLIC VARP _Im2Col(VARP x, INTS kernelSize, INTS dilate, INTS pads, INTS stride);
```
我们沿着原始矩阵逐行计算，将得到的新的子矩阵展开成列，放置在列块矩阵中


参数：
- `x` 输入变量
- `kernelSize` 内核大小
- `dilate` 扩张操作：控制kernel点的间距
- `pads` 填充操作
- `stride` 步长

返回：列块矩阵

---
### _Col2Im
```cpp
MNN_PUBLIC VARP _Col2Im(VARP x, VARP outputShape, INTS kernelSize, INTS dilate, INTS pads, INTS stride);
```
我们沿着列块矩阵逐行计算，将得到的行展成子矩阵，然后将子矩阵放置在最终结果对应的位置（每次当前值进行相加），同时记录每个位置的值放置的次数。最后，将当前位置的值除以放置的次数，即可得到结果（原始矩阵）


参数：
- `x` 输入变量
- `outputShape` 输出形状
- `kernelSize` 内核大小
- `dilate` 扩张操作：控制kernel点的间距
- `pads` 填充操作
- `stride` 步长

返回：原始矩阵
















