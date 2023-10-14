<!-- pymnn/CVImageProcess.md -->
## MNN.CVImageProcess *[deprecated]*

```python
class CVImageProcess
```
CVImageProcess用于图像处理，该图像处理类提供了一下图像处理能力：
- 图像格式转换，类似于`cv2.cvtColor`，通过设置`sourceFormat`和`destFormat`来实现
- 图像数据类型转换，将`uint8`类型的图像转换为`float32`类型的数据
- 图像的仿射变换，类似于`cv2.resize`和`cv2.warpAffine`，通过设置[CVMatrix](CVMatrix.md) 来实现
- 对图像进行归一化，通过设置`mean`和`normal`来实现; `x = (x - mean) / normal`

*不建议使用该接口，请使用[cv](cv.md)代替*
---
### `MNN.CV_ImageFormat_*`
描述图像格式的数据类型，支持RBG,RGBA,BGR,BGRA,GRAY,YUV_NV21类型
- 类型：`int`
- 枚举值：
  - `CV_ImageFormat_BGR`
  - `CV_ImageFormat_BGRA`
  - `CV_ImageFormat_RGB`
  - `CV_ImageFormat_RGBA`
  - `CV_ImageFormat_GRAY`
  - `CV_ImageFormat_YUV_NV21`

---
### `MNN.CV_Filter_*`
描述图片变换时的插值类型，支持最近邻，双线性，双三次插值
- 类型：`int`
- 枚举值：
  - `CV_Filter_NEAREST`
  - `CV_Filter_BILINEAL`
  - `CV_Filter_BICUBIC`

---
### `MNN.CV_Wrap_*`
描述图片变换时的填充方式，支持填0，重复，和最近值填充
- 类型：`int`
- 枚举值：
  - `CV_Wrap_ZERO`
  - `CV_Wrap_REPEAT`
  - `CV_Wrap_CLAMP_TO_EDGE`

---
### `CVImageProcess(config)`
根据config创建一个图像处理类

参数：
- `config:dict` 一个字典，其中的key和value的含义如表格所示

|    key        |   value    |              说明               |
|:--------------|:-----------|---------------------------------|
|   `filterType`   | `MNN.CV_Filter_*` | 用于进行图像缩放的滤波类型，默认为：`CV_Filter_NEAREST` |
|   `sourceFormat` | `MNN.CV_ImageFormat_*` | 用于对转换数据的数据格式进行定义，默认为：`CV_ImageFormat_BGRA` |
|   `destFormat` | `MNN.CV_ImageFormat_*` | 用于对转换数据的数据格式进行定义，默认为：`CV_ImageFormat_BGRA` |
|   `wrap` | `MNN.CV_Wrap_*` | 用于对转换后的图像进行填充，默认为：`CV_Wrap_ZERO` |
|   `mean` |  `tuple` | 用于对输入图像进行减均值处理，默认为：`(0, 0, 0, 0)`|
|   `normal` | `tuple` | 用于对输入图像进行归一化处理，默认为：`(1, 1, 1, 1)`|

返回：CVImageProcess对象

返回类型：`CVImageProcess`

---
### `setMatrix(matrix)`

设置仿射变换矩阵

参数：
- `matrix:CVMatrix` 图片仿射变换的变换矩阵, 参考[CVMatrix](CVMatrix.md)

返回：`None`

返回类型：`None`

---
### `setPadding(value)`

当填充类型为`CV_Wrap_ZERP`时，设置填充值，如果不设置则填充0

参数：
- `value:int` 填充值，默认填充0

返回：`None`

返回类型：`None`

---
### `convert(src, iw, ih, stride, dst)`

执行图像处理流程，将src中的数据按照config和matrix的要求进行转换，并将结果存入dst中

参数：
- `src:int|PyCapsule|tuple|ndarray` 输入的图像数据，可以是指针(int, PyCapsule)，也可以是数据（Tuple, ndarray）
- `iw:int` 输入图像的宽度
- `ih:int` 输入图像的高度
- `stride:int` 输入图像的步长，指每行的字节数，输入`0`则`stride=iw * ichannel`; *注意在处理YUV图像的时候必须传入stride*
- `dst:Tensor` 输出的图像Tensor

返回：`None`

返回类型：`None`

---
### `createImageTensor(dtype, width, height, channel, data)`

创建一个存储图像的Tensor

*该解口功能不完善，不建议使用*

参数：
- `dtype:MNN.Halide_Type_*` Tensor的数据类型
- `width:int` 图像的宽度
- `height:int` 图像的高度
- `channel:int` 图像的通道数
- `data:NoneType` 未使用参数 

返回：存储图像的Tensor对象

返回类型：`Tensor`

---
### `Example`

更多用法请参考[CVMatrix中的Example](CVMatrix.html#example)

```python
import MNN
import MNN.cv as cv

image = cv.imread('cat.jpg')
image_data = image.ptr
src_height, src_width, channel = image.shape
dst_height = dst_width = 224

# 对读入图像执行一下变换：
# 1. 图像格式转换：RGB -> BGR
# 2. 图像大小缩放：h,w -> 224,224
# 3. 图像类型变换：uint8 -> float32
# 4. 归一化处理：[0,255] -> [0,1]
dst_tensor = MNN.Tensor((1, dst_height, dst_width, channel), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Tensorflow)
image_processer = MNN.CVImageProcess({'sourceFormat': MNN.CV_ImageFormat_BGR,
                                      'destFormat': MNN.CV_ImageFormat_RGB,
                                      'mean': (127.5, 127.5, 127.5, 0),
                                      'filterType': MNN.CV_Filter_BILINEAL,
                                      'normal': (0.00784, 0.00784,0.00784, 1)})
#设置图像变换矩阵
matrix = MNN.CVMatrix()
x_scale = src_width / dst_width
y_scale = src_height / dst_height
matrix.setScale(x_scale, y_scale)
image_processer.setMatrix(matrix)
image_processer.convert(image_data, src_width, src_height, 0, dst_tensor)
```