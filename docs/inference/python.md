# Python API使用

## 安装
MNN Python API可以使用[源码安装](../compile/pymnn.md)，也可以直接使用`pip`安装预编译whl包；`pip`安装用法如下：
```bash
# 外部版本安装
pip install MNN==$version
# 公司内部版本安装
pip install -i https://artifacts.antgroup-inc.cn/simple/ MNN-Internal==$version
```

## 概览
MNN在C++的基础上，增加了Python扩展。扩展单元包括两个部分：
- MNN：负责推理，训练，图像处理和数值计算
- MNNTools：对MNN的部分工具进行封装，包括：mnn，mnnconvert和mnnquant

### MNN
[MNN](../pymnn/MNN.md)
- [expr](../pymnn/expr.md)
- [nn](../pymnn/nn.md)
  - [loss](../pymnn/loss.md)
  - [compress](../pymnn/compress.md)
- [data](../pymnn/data.md)
- [optim](../pymnn/optim.md)
- [cv](../pymnn/cv.md)
- [numpy](../pymnn/numpy.md)
  - [linalg](../pymnn/linalg.md)
  - [random](../pymnn/random.md)
### MNNTools
MNNTools提供目前主要是2个工具，用法可以参考[mnnconvert](../tools/python.html#mnnconvert)和[mnnquant](../tools/python.html#mnnquant)

## 使用Python Module API
### 数据类型
Python中的`Module API`与C++中的函数名略有区别，用法相似。主要数据类型如下：
- [_Module](../pymnn/_Module.md) 模型实例
- [Var](../pymnn/Var.md) 模型的输入输出
### 推理流程
基本推理流程如下：
- [创建Module](../pymnn/nn.html#load-module-from-file-file-name-input-names-output-names-dynamic-shape-mutable-rearrange-backend-memory-mode-power-mode-precision-mode)
- 创建输入: 使用`expr`或`numpy`函数创建`Var`即可作为输入
- [执行推理](../pymnn/_Module.html#forward-input)
- 获取输出: 输出为`Var`类型，可以通过`expr`或`numpy`函数执行后处理
### 示例
```python
import MNN.nn as nn
import MNN.cv as cv
import MNN.numpy as np
import MNN.expr as expr

# 配置执行后端，线程数，精度等信息；key-vlaue请查看API介绍
config = {}
config['precision'] = 'low' # 当硬件支持（armv8.2）时使用fp16推理
config['backend'] = 0       # CPU
config['numThread'] = 4     # 线程数

rt = nn.create_runtime_manager((config,))
# 加载模型创建_Module
net = nn.load_module_from_file('mobilenet_v1.mnn', ['data'], ['prob'], runtime_manager=rt)

# 读取图片
image = cv.imread('cat.jpg')
# 转换为float32, 形状为[224,224,3]
image = cv.resize(image, (224, 224), mean=[103.94, 116.78, 123.68], norm=[0.017, 0.017, 0.017])
# 增加batch HWC to NHWC
input_var = np.expand_dims(image, 0)
# NHWC to NC4HW4
input_var = expr.convert(input_var, expr.NC4HW4)

# 执行推理
output_var = net.forward(input_var)

# NC4HW4 to NHWC
output_var = expr.convert(output_var, expr.NHWC)
# 打印出分类结果, 282为猫
print("output belong to class: {}".format(np.argmax(output_var)))
# output belong to class: 282
```
其他示例可以参考[示例](../pymnn/RuntimeManager.html#example)；也可以参考[示例工程](../start/demo.html#id5)。


## 使用Python Session API *[deprecated]*

不建议使用该API执行推理，建议使用Module API

### 数据类型
Python中`Session API`的函数名与用法与C++基本一样。使用的主要数据类型如下：
- [Interpreter](../pymnn/Interpreter.md) 解释器，持有模型资源
- [Session](../pymnn/Session.md) 会话，持有推理资源
- [Tensor](../pymnn/Tensor.md) 用来描述输入输出数据
- [CVImageProcess](../pymnn/CVImageProcess.md) 图像处理模块
- [CVMatrix](../pymnn/CVMatrix.md) 用来描述图像的仿射变换
### 推理流程
基本推理流程如下：
- [创建Interpreter](../pymnn/Interpreter.html#interpreter-model-path)
- [创建Session](../pymnn/Interpreter.html#createsession-config-runtime)
- [获取Session的输入输出](../pymnn/Interpreter.html#getsessioninput-session-tensorname)
- [使用ImageProcess/cv进行图像处理（可选）](../pymnn/CVImageProcess.html#example)
- [拷贝数据到输入Tensor](../pymnn/Tensor.html#copyfrom-from)
- [执行resize（可选）](../pymnn/Interpreter.html#resizesession-session)
- [执行Session](../pymnn/Interpreter.html#runsession-session)
- [获取输出Tensor数据](../pymnn/Tensor.html#copytohosttensor-to)
### 示例
```python
import MNN
import MNN.cv as cv
import MNN.numpy as np
import MNN.expr as expr

# 创建interpreter
interpreter = MNN.Interpreter("mobilenet_v1.mnn")
# 创建session
config = {}
config['precision'] = 'low'
config['backend'] = 'CPU'
config['thread'] = 4
session = interpreter.createSession(config)
# 获取会话的输入输出
input_tensor = interpreter.getSessionInput(session)
output_tensor = interpreter.getSessionOutput(session)

# 读取图片
image = cv.imread('cat.jpg')

dst_height = dst_width = 224
# 使用ImageProcess处理第一张图片，将图片转换为转换为size=(224, 224), dtype=float32，并赋值给input_data1
image_processer = MNN.CVImageProcess({'sourceFormat': MNN.CV_ImageFormat_BGR,
                                      'destFormat': MNN.CV_ImageFormat_BGR,
                                      'mean': (103.94, 116.78, 123.68, 0.0),
                                      'filterType': MNN.CV_Filter_BILINEAL,
                                      'normal': (0.017, 0.017, 0.017, 0.0)})
image_data = image.ptr
src_height, src_width, channel = image.shape
input_data1 = MNN.Tensor((1, dst_height, dst_width, channel), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Tensorflow)
#设置图像变换矩阵
matrix = MNN.CVMatrix()
x_scale = src_width / dst_width
y_scale = src_height / dst_height
matrix.setScale(x_scale, y_scale)
image_processer.setMatrix(matrix)
image_processer.convert(image_data, src_width, src_height, 0, input_data1)

# 使用cv模块处理第二张图片，将图片转换为转换为size=(224, 224), dtype=float32，并赋值给input_data2
image = cv.imread('TestMe.jpg')
image = cv.resize(image, (224, 224), mean=[103.94, 116.78, 123.68], norm=[0.017, 0.017, 0.017])
input_data2 = np.expand_dims(image, 0) # [224, 224, 3] -> [1, 224, 224, 3]

# 合并2张图片到，并赋值给input_data
input_data1 = expr.const(input_data1.getHost(), input_data1.getShape(), expr.NHWC) # Tensor -> Var
input_data = np.concatenate([input_data1, input_data2])  # [2, 224, 224, 3]
input_data = MNN.Tensor(input_data) # Var -> Tensor

# 演示多张图片输入，所以将输入resize到[2, 3, 224, 224]
interpreter.resizeTensor(input_tensor, (2, 3, 224, 224))
# 重新计算形状分配内存
interpreter.resizeSession(session)

# 拷贝数据到输入Tensor
input_tensor.copyFrom(input_data)

# 执行会话推理
interpreter.runSession(session)

# 从输出Tensor拷贝出数据 
output_data = MNN.Tensor(output_tensor.getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
output_tensor.copyToHostTensor(output_data)

# 打印出分类结果: 282为猫，385为象
output_var = expr.const(output_data.getHost(), [2, 1001])
print("output belong to class: {}".format(np.argmax(output_var, 1)))
# output belong to class: array([282, 385], dtype=int32)
```
其他示例可以参考[示例](../pymnn/Interpreter.html#example)；也可以参考[示例工程](../start/demo.html#session)。

## 使用cv/numpy API
### 数据类型
Python的`cv`和`numpy`接口，其中`cv`是对C++中`tools/cv`实现的封装；`numpy`则是对`expr`接口的封装；这两个接口主要为了提高MNN的易用性，与`opencv`与`numpy`做到了再接口上的部分兼容，在用法和思路上基本一致。主要数据类型如下：
- [Var](../pymnn/Var.md) `cv`中的图像，`numpy`中的`ndarray`
### 主要用法
`cv`和`numpy`主要用作模型的前后处理部分，和一些数值计算任务。比如从图片直接读取数据后一般需要执行颜色空间变换，数据类型变换，缩放，裁剪等操作，这些可以用`cv`模块函数实现；模型输出的结果可能需要做一些额外的变换和计算，这些可以用`numpy`模块函数实现。
### 示例
使用`cv`与`numpy`中的函数做前后处理，执行模型推理的例子
```python
import MNN
import MNN.cv as cv
import MNN.numpy as np

# 加载模型
net = MNN.nn.load_module_from_file('mobilenet_v1.mnn', ["data"], ["prob"])
# cv模块图片处理
image = cv.imread('cat.jpg')
image = cv.resize(image, (224, 224))
# 类似ndarray的数值运算
image = image - (103.94, 116.78, 123.68)
image = image * (0.017, 0.017, 0.017)
input_var = np.expand_dims(image, 0)
input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
output_var = net.forward(input_var)
output_var = MNN.expr.convert(output_var, MNN.expr.NHWC)
# 类似numpy操作的后处理
print("output belong to class: {}".format(np.argmax(output_var)))
```
其他示例可以参考[示例](../pymnn/Interpreter.html#example)；也可以参考[示例工程](../start/demo.html#id5)。

### cv能力列表
cv模块提供了与OpenCV相似的接口函数，具备基础的图像处理能力，目前支持的cv函数60个。

#### 图像编解码

| 函数名 | 功能 |
| --- | --- |
| [haveImageReader](../pymnn/cv.html#haveimagereader-filename) | 是否可读（解码） |
| [haveImageWriter](../pymnn/cv.html#haveimagewriter-filename) | 是否可写（编码） |
| [imdecode](../pymnn/cv.html#imdecode-buf-flag) | 从内存解码为Mat |
| [imencode](../pymnn/cv.html#imencode-ext-img-params) | 编码Mat到内存中 |
| [imread](../pymnn/cv.html#imread-filename-flag) | 读图片 |
| [imwrite](../pymnn/cv.html#imwrite-filename-img-params) | 写图片 |

#### 图像滤波

| 函数名 | 功能 |
| --- | --- |
| [blur](../pymnn/cv.html#blur-src-ksize-bordertype) | 均值滤波，平滑模糊 |
| [boxFilter](../pymnn/cv.html#boxfilter-src-ddepth-ksize-normalize-bordertype) | 盒子滤波， |
| [dilate](../pymnn/cv.html#dilate-src-kernel-iterations-bordertype) | 膨胀 |
| [filter2D](../pymnn/cv.html#filter2d-src-ddepth-kernel-delta-bordertype) | 2d卷积 |
| [GaussianBlur](../pymnn/cv.html#gaussianblur-src-ksize-sigmax-sigmay-bordertype) | 高斯模糊 |
| [getDerivKernels](../pymnn/cv.html#getderivkernels-dx-dy-ksize-normalize) | 求导数，实际为Sobel/Scharr |
| [getGaborKernel](../pymnn/cv.html#getgaborkernel-ksize-sigma-theta-lambd-gamma-psi) | 获取Gabor核 |
| [getGaussianKernel](../pymnn/cv.html#getgaussiankernel-ksize-sigma) | 获得高斯核 |
| [getStructuringElement](../pymnn/cv.html#getstructuringelement-shape-ksize) | 获取结构化元素用于形态学操作 |
| [Laplacian](../pymnn/cv.html#laplacian-src-ddepth-ksize-scale-delta-bordertype) | 边缘检测滤波 |
| [pyrDown](../pymnn/cv.html#pyrdown-src-dstsize-bordertype) | 高斯平滑+下采样 |
| [pyrUp](../pymnn/cv.html#pyrup-src-dstsize-bordertype) | 上采样+高斯平滑 |
| [Scharr](../pymnn/cv.html#scharr-src-ddepth-dx-dy-scale-delta-bordertype) | 边缘检测滤波 |
| [sepFilter2D](../pymnn/cv.html#sepfilter2d-src-ddepth-kx-ky-delta-bordertype) | 2个一维kernel做滤波 |
| [Sobel](../pymnn/cv.html#sobel-src-ddepth-dx-dy-ksize-scale-delta-bordertype) | 边缘检测滤波 |
| [spatialGradient](../pymnn/cv.html#spatialgradient-src-ksize-bordertype) | 梯度，实际为Sobel |
| [sqrBoxFilter](../pymnn/cv.html#sqrboxfilter-src-ddepth-ksize-normalize-bordertype) | 平方后滤波 |

#### 图像形变

| 函数名 | 功能 |
| --- | --- |
| [getAffineTransform](../pymnn/cv.html#getaffinetransform-src-dst) | 仿射变换 |
| [getPerspectiveTransform](../pymnn/cv.html#getperspectivetransform-src-dst) | 透视变换 |
| [getRectSubPix](../pymnn/cv.html#getrectsubpix-image-patchsize-center) | 截取矩形区域 |
| [getRotationMatrix2D](../pymnn/cv.html#getrotationmatrix2d-center-angle-scale) | 旋转矩阵 |
| [invertAffineTransform](../pymnn/cv.html#invertaffinetransform-m) | 仿射变换矩阵求逆 |
| [resize](../pymnn/cv.html#resize-src-dsize-fx-fy-interpolation-code-mean-norm) | 图片放缩 |
| [warpAffine](../pymnn/cv.html#warpaffine-src-m-dsize-flag-bordermode-bordervalue-code-mean-norm) | 仿射变换 |
| [warpPerspective](../pymnn/cv.html#warpperspective-src-m-dsize-flag-bordermode-bordervalue) | 透视变换 |

#### 图像转换

| 函数名 | 功能 |
| --- | --- |
| [blendLinear](../pymnn/cv.html#blendlinear-src1-src2-weight1-weight2) | 线性混合2个图像 |
| [threshold](../pymnn/cv.html#threshold-src-thresh-maxval-type) | 逐像素阈值化 |

#### 绘画函数

| 函数名 | 功能 |
| --- | --- |
| [arrowedLine](../pymnn/cv.html#arrowedline-img-pt1-pt2-color-thickness-linetype-shift-tiplength) | 画箭头 |
| [circle](../pymnn/cv.html#circle-img-center-radius-color-thickness-linetype-shift) | 画圆 |
| [drawContours](../pymnn/cv.html#drawcontours-img-contours-contouridx-color-thickness-linetype) | 画轮廓 |
| [fillPoly](../pymnn/cv.html#fillpoly-img-contours-color-linetype-shift-offset) | 填充多边形 |
| [line](../pymnn/cv.html#line-img-pt1-pt2-color-thickness-linetype-shift) | 画线段 |
| [rectangle](../pymnn/cv.html#rectangle-src-pt1-pt2-color-thickness-linetype-shift) | 画正方向 |

#### 色彩空间转换

| 函数名 | 功能 |
| --- | --- |
| [cvtColor](../pymnn/cv.html#cvtcolor-src-code-dstcn) | 颜色空间转换 |
| [cvtColorTwoPlane](../pymnn/cv.html#cvtcolortwoplane-src1-src2-code) | YUV420到RGB的转换 |

#### 结构函数

| 函数名 | 功能 |
| --- | --- |
| [findContours](../pymnn/cv.html#findcontours-image-mode-method-offset) | 轮廓检测 |
| [contourArea](../pymnn/cv.html#contourarea-points-oriented) | 计算轮廓的面积 |
| [convexHull](../pymnn/cv.html#convexhull-points-clockwise-returnpoints) | 计算点集的凸包 |
| [minAreaRect](../pymnn/cv.html#minarearect-points) | 最小外接矩形 |
| [boundingRect](../pymnn/cv.html#boundingrect-points) | 计算点集的最小外接矩形 |
| [connectedComponentsWithStats](../pymnn/cv.html#connectedcomponentswithstats-image-connectivity) | 计算图像的连通域 |
| [boxPoints](../pymnn/cv.html#boxpoints-box) | 计算矩形的四个顶点坐标 |

#### 直方图

| 函数名 | 功能 |
| --- | --- |
| [calcHist](../pymnn/cv.html#calchist-imgs-channels-mask-histsize-ranges-accumulate) | 计算直方图 |

#### 3D

| 函数名 | 功能 |
| --- | --- |
| [Rodrigues](../pymnn/cv.html#rodrigues-src) | 旋转矩阵转换为旋转向量 |
| [solvePnP](../pymnn/cv.html#solvepnp-objectpoints-imagepoints-cameramatrix-distcoeffs-flags) | 计算2d到3d的映射 |

#### 数组操作函数

| 函数名 | 功能 |
| --- | --- |
| [copyTo](../pymnn/cv.html#copyto-src-mask-dst) | 带mask的拷贝 |
| [bitwise_and](../pymnn/cv.html#bitwise-and-src1-src2-dst-mask) | 带mask按位与 |
| [bitwise_or](../pymnn/cv.html#bitwise-or-src1-src2-dst-mask) | 带mask按位或 |
| [bitwise_xor](../pymnn/cv.html#bitwise-xor-src1-src2-dst-mask) | 带mask按位异或 |
| [hconcat](../pymnn/cv.html#hconcat-src) | 水平方向拼接 |
| [vconcat](../pymnn/cv.html#vconcat-src) | 垂直方向拼接 |
| [mean](../pymnn/cv.html#mean-src-mask) | 求均值 |
| [flip](../pymnn/cv.html#flip-src-flipcode) | 翻转 |
| [rotate](../pymnn/cv.html#rotate-src-rotatemode) | 旋转 |

### numpy能力列表
numpy函数170个，函数列表如下:
#### 数组创建

| 函数名 | 功能 |
| --- | --- |
| [empty](../numpy.html#empty-shape-dtype-float32-order-c) | 空数组 |
| [empty_like](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html#numpy.empty_like) | 空数组like |
| [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html#numpy.eye) | 对角线2d数组 |
| [identity](https://numpy.org/doc/stable/reference/generated/numpy.identity.html#numpy.identity) | 对角线2d数组 |
| [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html#numpy.ones) | 全1数组 |
| [ones_like](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html#numpy.ones_like) | 全1数组like |
| [zeros](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html#numpy.zeros) | 全0数组 |
| [zeros_like](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html#numpy.zeros_like) | 全0数组like |
| [full](https://numpy.org/doc/stable/reference/generated/numpy.full.html#numpy.full) | 填充 |
| [full_like](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html#numpy.full_like) | 填充like |
| [array](https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array) | 创建数组 |
| [asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy.asarray) | 创建数组 |
| [asanyarray](https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html#numpy.asanyarray) | 创建数组 |
| [ascontiguousarray](https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray) | 创建数组 |
| [asmatrix](https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html#numpy.asmatrix) | 创建2d数组 |
| [copy](https://numpy.org/doc/stable/reference/generated/numpy.copy.html#numpy.copy) | 拷贝数组 |
| [arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy.arange) | 范围创建 |
| [linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace) | 区间创建 |
| [logspace](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html#numpy.logspace) | log区间创建 |
| [geomspace](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html#numpy.geomspace) | log区间创建 |
| [meshgrid](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html#numpy.meshgrid) | 坐标矩阵 |
| [mat](https://numpy.org/doc/stable/reference/generated/numpy.mat.html#numpy.mat) | 矩阵 |

#### 数组操作

| 函数名 | 功能 |
| --- | --- |
| [copyto](https://numpy.org/doc/stable/reference/generated/numpy.copyto.html#numpy.copyto) | 拷贝至 |
| [shape](https://numpy.org/doc/stable/reference/generated/numpy.shape.html#numpy.shape) | 获取形状 |
| [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape) | 改变形状 |
| [ravel](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html#numpy.ravel) | 拉平 |
| [flat](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat) | 拉平 |
| [flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten) | 拉平 |
| [moveaxis](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html#numpy.moveaxis) | 移动维度 |
| [rollaxis](https://numpy.org/doc/stable/reference/generated/numpy.rollaxis.html#numpy.rollaxis) | 轮转维度 |
| [swapaxes](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html#numpy.swapaxes) | 交换维度 |
| [T](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T) | 转置 |
| [transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html#numpy.transpose) | 转置 |
| [atleast_1d](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d) | 至少1维 |
| [atleast_2d](https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d) | 至少2维 |
| [atleast_3d](https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d) | 至少3维 |
| [broadcast_to](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to) | 广播 |
| [broadcast_arrays](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html#numpy.broadcast_arrays) | 数组广播 |
| [expand_dims](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html#numpy.expand_dims) | 增加维度 |
| [squeeze](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html#numpy.squeeze) | 压缩1维度 |
| [asfarray](https://numpy.org/doc/stable/reference/generated/numpy.asfarray.html#numpy.asfarray) | 转浮点 |
| [asscalar](https://numpy.org/doc/stable/reference/generated/numpy.asscalar.html#numpy.asscalar) | 转标量 |
| [concatenate](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html#numpy.concatenate) | 连接 |
| [stack](https://numpy.org/doc/stable/reference/generated/numpy.stack.html#numpy.stack) | 连接 |
| [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy.vstack) | 垂直连接 |
| [hstack](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html#numpy.hstack) | 水平连接 |
| [dstack](https://numpy.org/doc/stable/reference/generated/numpy.dstack.html#numpy.dstack) | 深度连接 |
| [column_stack](https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html#numpy.column_stack) | 列连接 |
| [row_stack](https://numpy.org/doc/stable/reference/generated/numpy.row_stack.html#numpy.row_stack) | 行连接 |
| [split](https://numpy.org/doc/stable/reference/generated/numpy.split.html#numpy.split) | 切分 |
| [array_split](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split) | 数组切分 |
| [dsplit](https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html#numpy.dsplit) | 深度切分 |
| [hsplit](https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html#numpy.hsplit) | 水平切分 |
| [vsplit](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html#numpy.vsplit) | 垂直切分 |
| [tile](https://numpy.org/doc/stable/reference/generated/numpy.tile.html#numpy.tile) | 重复堆叠 |
| [repeat](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat) | 重复 |
| [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape) | 变形 |

#### 坐标操作

| 函数名 | 功能 |
| --- | --- |
| [nonzero](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html#numpy.nonzero) | 非0元素坐标 |
| [where](https://numpy.org/doc/stable/reference/generated/numpy.where.html#numpy.where) | 条件选取 |
| [unravel_index](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html#numpy.unravel_index) | 反拉平坐标 |

#### 线性代数

| 函数名 | 功能 |
| --- | --- |
| [dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy.dot) | 点乘 |
| [vdot](https://numpy.org/doc/stable/reference/generated/numpy.vdot.html#numpy.vdot) | 点乘 |
| [inner](https://numpy.org/doc/stable/reference/generated/numpy.inner.html#numpy.inner) | 内积 |
| [matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul) | 矩阵乘 |

#### 逻辑函数

| 函数名 | 功能 |
| --- | --- |
| [all](https://numpy.org/doc/stable/reference/generated/numpy.all.html#numpy.all) | 全部非0 |
| [any](https://numpy.org/doc/stable/reference/generated/numpy.any.html#numpy.any) | 任意非0 |
| [logical_and](https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html#numpy.logical_and) | 与 |
| [logical_or](https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html#numpy.logical_or) | 或 |
| [logical_not](https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html#numpy.logical_not) | 否 |
| [logical_xor](https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html#numpy.logical_xor) | 异或 |
| [array_equal](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html#numpy.array_equal) | 相等 |
| [array_equiv](https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html#numpy.array_equiv) | 相等 |
| [greater](https://numpy.org/doc/stable/reference/generated/numpy.greater.html#numpy.greater) | 大于 |
| [greater_equal](https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html#numpy.greater_equal) | 大于等于 |
| [less](https://numpy.org/doc/stable/reference/generated/numpy.less.html#numpy.less) | 小于 |
| [less_equal](https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html#numpy.less_equal) | 小于等于 |
| [equal](https://numpy.org/doc/stable/reference/generated/numpy.equal.html#numpy.equal) | 等于 |
| [not_equal](https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html#numpy.not_equal) | 不等 |

#### 数学函数

| API | 功能 |
| --- | --- |
| [sin](https://numpy.org/doc/stable/reference/generated/numpy.sin.html#numpy.sin) | 正弦 |
| [cos](https://numpy.org/doc/stable/reference/generated/numpy.cos.html#numpy.cos) | 余弦 |
| [tan](https://numpy.org/doc/stable/reference/generated/numpy.tan.html#numpy.tan) | 正切 |
| [arcsin](https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html#numpy.arcsin) | 反正弦 |
| [arccos](https://numpy.org/doc/stable/reference/generated/numpy.arccos.html#numpy.arccos) | 反余弦 |
| [arctan](https://numpy.org/doc/stable/reference/generated/numpy.arctan.html#numpy.arctan) | 反正切 |
| [hypot](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html#numpy.hypot) | 
 |
| [arctan2](https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html#numpy.arctan2) | 
 |
| [sinh](https://numpy.org/doc/stable/reference/generated/numpy.sinh.html#numpy.sinh) | 
 |
| [cosh](https://numpy.org/doc/stable/reference/generated/numpy.cosh.html#numpy.cosh) | 
 |
| [tanh](https://numpy.org/doc/stable/reference/generated/numpy.tanh.html#numpy.tanh) | 
 |
| [arcsinh](https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html#numpy.arcsinh) | 
 |
| [arccosh](https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html#numpy.arccosh) | 
 |
| [arctanh](https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html#numpy.arctanh) | 
 |
| [around](https://numpy.org/doc/stable/reference/generated/numpy.around.html#numpy.around) | 
 |
| [round_](https://numpy.org/doc/stable/reference/generated/numpy.round_.html#numpy.round_) | 
 |
| [rint](https://numpy.org/doc/stable/reference/generated/numpy.rint.html#numpy.rint) | 
 |
| [floor](https://numpy.org/doc/stable/reference/generated/numpy.floor.html#numpy.floor) | 
 |
| [ceil](https://numpy.org/doc/stable/reference/generated/numpy.ceil.html#numpy.ceil) | 
 |
| [trunc](https://numpy.org/doc/stable/reference/generated/numpy.trunc.html#numpy.trunc) | 
 |
| [prod](https://numpy.org/doc/stable/reference/generated/numpy.prod.html#numpy.prod) | 积 |
| [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum) | 和 |
| [nanprod](https://numpy.org/doc/stable/reference/generated/numpy.nanprod.html#numpy.nanprod) | 积 |
| [nansum](https://numpy.org/doc/stable/reference/generated/numpy.nansum.html#numpy.nansum) | 和 |
| [exp](https://numpy.org/doc/stable/reference/generated/numpy.exp.html#numpy.exp) | e指数 |
| [expm1](https://numpy.org/doc/stable/reference/generated/numpy.expm1.html#numpy.expm1) | e指数-1 |
| [exp2](https://numpy.org/doc/stable/reference/generated/numpy.exp2.html#numpy.exp2) | 2指数 |
| [log](https://numpy.org/doc/stable/reference/generated/numpy.log.html#numpy.log) | 对数 |
| [log10](https://numpy.org/doc/stable/reference/generated/numpy.log10.html#numpy.log10) | 10对数 |
| [log2](https://numpy.org/doc/stable/reference/generated/numpy.log2.html#numpy.log2) | 2对数 |
| [log1p](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html#numpy.log1p) | x+1对数 |
| [logaddexp](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html#numpy.logaddexp) | exp对数 |
| [logaddexp2](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html#numpy.logaddexp2) | 2指数对数 |
| [sinc](https://numpy.org/doc/stable/reference/generated/numpy.sinc.html#numpy.sinc) |  |
| [signbit](https://numpy.org/doc/stable/reference/generated/numpy.signbit.html#numpy.signbit) |  |
| [copysign](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html#numpy.copysign) |  |
| [frexp](https://numpy.org/doc/stable/reference/generated/numpy.frexp.html#numpy.frexp) |  |
| [ldexp](https://numpy.org/doc/stable/reference/generated/numpy.ldexp.html#numpy.ldexp) |  |
| [add](https://numpy.org/doc/stable/reference/generated/numpy.add.html#numpy.add) | 加 |
| [reciprocal](https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html#numpy.reciprocal) | 倒数 |
| [positive](https://numpy.org/doc/stable/reference/generated/numpy.positive.html#numpy.positive) | 取正 |
| [negative](https://numpy.org/doc/stable/reference/generated/numpy.negative.html#numpy.negative) | 取负 |
| [multiply](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html#numpy.multiply) | 乘 |
| [divide](https://numpy.org/doc/stable/reference/generated/numpy.divide.html#numpy.divide) | 除 |
| [power](https://numpy.org/doc/stable/reference/generated/numpy.power.html#numpy.power) | 指数 |
| [subtract](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html#numpy.subtract) | 减 |
| [true_divide](https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html#numpy.true_divide) | 除 |
| [floor_divide](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html#numpy.floor_divide) | 除 |
| [float_power](https://numpy.org/doc/stable/reference/generated/numpy.float_power.html#numpy.float_power) | 指数 |
| [fmod](https://numpy.org/doc/stable/reference/generated/numpy.fmod.html#numpy.fmod) | 模 |
| [mod](https://numpy.org/doc/stable/reference/generated/numpy.mod.html#numpy.mod) | 模 |
| [modf](https://numpy.org/doc/stable/reference/generated/numpy.modf.html#numpy.modf) | 模 |
| [remainder](https://numpy.org/doc/stable/reference/generated/numpy.remainder.html#numpy.remainder) | 余 |
| [divmod](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html#numpy.divmod) | 除，余 |
| [convolve](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html#numpy.convolve) | 卷积 |
| [clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html#numpy.clip) | 缩小范围 |
| [sqrt](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html#numpy.sqrt) | 平方根 |
| [cbrt](https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html#numpy.cbrt) | 立方根 |
| [square](https://numpy.org/doc/stable/reference/generated/numpy.square.html#numpy.square) | 平方 |
| [absolute](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html#numpy.absolute) | 绝对值 |
| [fabs](https://numpy.org/doc/stable/reference/generated/numpy.fabs.html#numpy.fabs) | 绝对值 |
| [sign](https://numpy.org/doc/stable/reference/generated/numpy.sign.html#numpy.sign) | 符号 |
| [maximum](https://numpy.org/doc/stable/reference/generated/numpy.maximum.html#numpy.maximum) | 取大 |
| [minimum](https://numpy.org/doc/stable/reference/generated/numpy.minimum.html#numpy.minimum) | 取小 |
| [fmax](https://numpy.org/doc/stable/reference/generated/numpy.fmax.html#numpy.fmax) | 取大 |
| [fmin](https://numpy.org/doc/stable/reference/generated/numpy.fmin.html#numpy.fmin) | 取小 |

#### 数组扩充

| 函数名 | 功能 |
| --- | --- |
| [pad](https://numpy.org/doc/stable/reference/generated/numpy.pad.html#numpy.pad) | 扩充 |

#### 随机采样

| 函数名 | 功能 |
| --- | --- |
| random | 随机数 |
| rand | 随机数 |
| randn | 随机数 |
| randint | 随机定点数 |

#### 排序，搜索，计数

| 函数名 | 功能 |
| --- | --- |
| [sort](https://numpy.org/doc/stable/reference/generated/numpy.sort.html#numpy.sort)，[lexsort](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html#numpy.lexsort)，[argsort](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html#numpy.argsort) | 排序 |
| [argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html#numpy.argmax) | 最大值坐标 |
| [nanargmax](https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html#numpy.nanargmax) | 最大值坐标 |
| [argmin](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html#numpy.argmin) | 最小值坐标 |
| [nanargmin](https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html#numpy.nanargmin) | 最小值坐标 |
| [argwhere](https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html#numpy.argwhere) | 非0坐标 |
| [flatnonzero](https://numpy.org/doc/stable/reference/generated/numpy.flatnonzero.html#numpy.flatnonzero) | 非0元素 |
| [count_nonzero](https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero) | 非0总数 |

#### 统计

| 函数名 | 功能 |
| --- | --- |
| [amin](https://numpy.org/doc/stable/reference/generated/numpy.amin.html#numpy.amin) | 最小值 |
| [amax](https://numpy.org/doc/stable/reference/generated/numpy.amax.html#numpy.amax) | 最大值 |
| [nanmin](https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html#numpy.nanmin) | 最小值 |
| [nanmax](https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html#numpy.nanmax) | 最大值 |
| [ptp](https://numpy.org/doc/stable/reference/generated/numpy.ptp.html#numpy.ptp) | 范围 |
| [average](https://numpy.org/doc/stable/reference/generated/numpy.average.html#numpy.average) | 均值 |
| [mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html#numpy.mean) | 均值 |
| [std](https://numpy.org/doc/stable/reference/generated/numpy.std.html#numpy.std) | 标准差 |
| [var](https://numpy.org/doc/stable/reference/generated/numpy.var.html#numpy.var) | 方差 |
| [nanmean](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html#numpy.nanmean) | 均值 |
| [nanstd](https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html#numpy.nanstd) | 标准差 |
| [nanvar](https://numpy.org/doc/stable/reference/generated/numpy.nanvar.html#numpy.nanvar) | 方差 |
