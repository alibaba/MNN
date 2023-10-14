<!-- pymnn/CVMatrix.md -->
## MNN.CVMatrix *[deprecated]*

```python
class CVMatrix
```
CVMatrix主要用在[CVImageProcess](CVImageProcess.md)中使用;
CVMatrix是MNN图像处理中使用的类，用来描述仿射变化的矩阵值，其内部存储了一个3x3的float矩阵，并提供了基本的矩阵操作。
CVMatrix移植自Android系统使用的Skia引擎，主要用于设置从目标图像到源图像的变换矩阵, [SKia文档](https://api.skia.org/classSkMatrix.html#details)。


*使用中一定要注意， CVMatrix表示的是「目标图像」到「源图像」的变换矩阵，平时我们在编写代码过程中，直觉上写出的是将源图像转换到目标图像，因此需要注意设置中需要以逆序来进行*

---
### `CVMatrix()`
创建一个默认的CVMatrix，其值如下：

    | 1 0 0 |
    | 0 1 0 |
    | 0 0 1 |

参数：
- `None`

返回：CVMatrix对象

返回类型：`CVMatrix`

---
### `setScale(sx, sy, |px, py)`

Sets Matrix to scale by sx and sy, about a pivot point at (px, py). 

设置矩阵比例缩放，比例缩放的值为sx和sy，缩放的中心点为(px, py)。缩放中心点不受矩阵的影响。

参数：
- `sx:int/float` 水平缩放值
- `sy:int/float` 垂直缩放值
- `px:int/float` 中心点坐标，默认为`0`
- `py:int/float` 中心点坐标，默认为`0`

返回：`None`

返回类型：`None`

---
### `preScale(sx, sy, |px, py)`

Sets Matrix to Matrix multiplied by Matrix constructed from scaling by (sx, sy) about pivot point (px, py). 

设置矩阵为矩阵乘以仿射变换矩阵，其中仿射变换矩阵为比例缩放矩阵，比例缩放的值为sx和sy，缩放的中心点为(px, py)。

Given:

              | A B C |                       | sx  0 dx |
    Matrix =  | D E F |,  S(sx, sy, px, py) = |  0 sy dy |
              | G H I |                       |  0  0  1 |

where

    dx = px - sx * px
    dy = py - sy * py

sets Matrix to:

                                  | A B C | | sx  0 dx |   | A*sx B*sy A*dx+B*dy+C |
    Matrix * S(sx, sy, px, py) =  | D E F | |  0 sy dy | = | D*sx E*sy D*dx+E*dy+F |
                                  | G H I | |  0  0  1 |   | G*sx H*sy G*dx+H*dy+I |

参数：
- `sx:int/float` 水平缩放值
- `sy:int/float` 垂直缩放值
- `px:int/float` 中心点坐标，默认为`0`
- `py:int/float` 中心点坐标，默认为`0`

返回：`None`

返回类型：`None`

---
### `postScale(sx, sy, |px, py)`

Sets Matrix to Matrix constructed from scaling by (sx, sy) about pivot point (px, py), multiplied by Matrix.

设置矩阵为比例缩放矩阵，比例缩放的值为sx和sy，缩放的中心点为(px, py)，然后乘以矩阵。

Given:

              | J K L |                       | sx  0 dx |
    Matrix =  | M N O |,  S(sx, sy, px, py) = |  0 sy dy |
              | P Q R |                       |  0  0  1 |

where

    dx = px - sx * px
    dy = py - sy * py

sets Matrix to:

                                  | sx  0 dx | | J K L |   | sx*J+dx*P sx*K+dx*Q sx*L+dx+R |
    S(sx, sy, px, py) * Matrix =  |  0 sy dy | | M N O | = | sy*M+dy*P sy*N+dy*Q sy*O+dy*R |
                                  |  0  0  1 | | P Q R |   |         P         Q         R |

参数：
- `sx:int/float` 水平缩放值
- `sy:int/float` 垂直缩放值
- `px:int/float` 中心点坐标，默认为`0`
- `py:int/float` 中心点坐标，默认为`0`

返回：`None`

返回类型：`None`

---
### `setRotate(degrees, |px, py)`

Sets Matrix to rotate by degrees about a pivot point at (px, py). The pivot point is unchanged when mapped with Matrix. Positive degrees rotates clockwise.

设置矩阵为旋转矩阵，旋转的角度为degrees，旋转的中心点为(px, py)。缩放中心点不受矩阵的影响。角度为正时为顺时针旋转。

参数：
- `degrees:int/float` 旋转角度
- `px:int/float` 中心点坐标，默认为`0`
- `py:int/float` 中心点坐标，默认为`0`

返回：`None`

返回类型：`None`

---
### `preRotate(degrees, |px, py)`

Sets Matrix to Matrix multiplied by Matrix constructed from rotating by degrees about pivot point (px, py).

设置矩阵为矩阵乘以旋转矩阵，旋转的角度为degrees，旋转的中心点为(px, py)。

Given:

              | A B C |                        | c -s dx |
    Matrix =  | D E F |,  R(degrees, px, py) = | s  c dy |
              | G H I |                        | 0  0  1 |

where

    c  = cos(degrees)
    s  = sin(degrees)
    dx =  s * py + (1 - c) * px
    dy = -s * px + (1 - c) * py

sets Matrix to:

                                  | A B C | | c -s dx |   | Ac+Bs -As+Bc A*dx+B*dy+C |
    Matrix * R(degrees, px, py) = | D E F | | s  c dy | = | Dc+Es -Ds+Ec D*dx+E*dy+F |
                                  | G H I | | 0  0  1 |   | Gc+Hs -Gs+Hc G*dx+H*dy+I |

参数：
- `degrees:int/float` 旋转角度
- `px:int/float` 中心点坐标，默认为`0`
- `py:int/float` 中心点坐标，默认为`0`

返回：`None`

返回类型：`None`

---
### `postRotate(degrees, |px, py)`

Sets Matrix to Matrix constructed from rotating by degrees about pivot point (0, 0), multiplied by Matrix. 

设置矩阵为旋转矩阵，旋转的角度为degrees，旋转的中心点为(0, 0)，然后乘以矩阵。

Given:

              | J K L |                        | c -s dx |
    Matrix =  | M N O |,  R(degrees, px, py) = | s  c dy |
              | P Q R |                        | 0  0  1 |

where

    c  = cos(degrees)
    s  = sin(degrees)
    dx =  s * py + (1 - c) * px
    dy = -s * px + (1 - c) * py

sets Matrix to:

                                  |c -s dx| |J K L|   |cJ-sM+dx*P cK-sN+dx*Q cL-sO+dx+R|
    R(degrees, px, py) * Matrix = |s  c dy| |M N O| = |sJ+cM+dy*P sK+cN+dy*Q sL+cO+dy*R|
                                  |0  0  1| |P Q R|   |         P          Q          R|

参数：
- `degrees:int/float` 旋转角度
- `px:int/float` 中心点坐标，默认为`0`
- `py:int/float` 中心点坐标，默认为`0`

返回：`None`

返回类型：`None`

---
### `setTranslate(dx, dy)`

Sets Matrix to translate by (dx, dy).

设置矩阵为平移矩阵，平移的值为(dx, dy)。

参数：
- `dx:int/float` 水平平移值
- `dy:int/float` 垂直平移值

返回：`None`

返回类型：`None`

---
### `preTranslate(dx, dy)`

Sets Matrix to Matrix multiplied by Matrix constructed from translation (dx, dy).

设置矩阵为矩阵乘以平移矩阵，平移的值为(dx, dy)。

Given:

              | A B C |               | 1 0 dx |
    Matrix =  | D E F |,  T(dx, dy) = | 0 1 dy |
              | G H I |               | 0 0  1 |

sets Matrix to:

                          | A B C | | 1 0 dx |   | A B A*dx+B*dy+C |
    Matrix * T(dx, dy) =  | D E F | | 0 1 dy | = | D E D*dx+E*dy+F |
                          | G H I | | 0 0  1 |   | G H G*dx+H*dy+I |

参数：
- `dx:int/float` 水平平移值
- `dy:int/float` 垂直平移值

返回：`None`

返回类型：`None`

---
### `postTranslate(dx, dy)`

Sets Matrix to Matrix constructed from translation (dx, dy) multiplied by Matrix.
This can be thought of as moving the point to be mapped after applying Matrix. 

设置矩阵为平移矩阵，平移的值为(dx, dy)，然后乘以矩阵。

Given:

              | J K L |               | 1 0 dx |
    Matrix =  | M N O |,  T(dx, dy) = | 0 1 dy |
              | P Q R |               | 0 0  1 |

sets Matrix to:

                          | 1 0 dx | | J K L |   | J+dx*P K+dx*Q L+dx*R |
    T(dx, dy) * Matrix =  | 0 1 dy | | M N O | = | M+dy*P N+dy*Q O+dy*R |
                          | 0 0  1 | | P Q R |   |      P      Q      R |

参数：
- `dx:int/float` 水平平移值
- `dy:int/float` 垂直平移值

返回：`None`

返回类型：`None`

---
### `setPolyToPoly(src, dst)`

Sets Matrix to map src to dst. count must be zero or greater, and four or less. If count is zero, sets Matrix to identity and returns true. If count is one, sets Matrix to translate and returns true. If count is two or more, sets Matrix to map Point if possible; returns false if Matrix cannot be constructed. If count is four, Matrix may include perspective.

设置矩阵，将src的坐标列表映射到dst的坐标列表。坐标的数目需大于等于0，小于等于4。
- 如果数目为0，将矩阵设置为单位矩阵
- 如果数目为1，将矩阵设置为平移矩阵
- 如果数目大于等于2，将矩阵设置为映射坐标（有可能没有正确的映射）
- 如果数目为4，矩阵可能包含透视值

参数：
- `src:[float]` 源点坐标列表
- `dst:[float]` 目标点坐标列表

返回：`None`

返回类型：`None`

---
### `invert()`

Sets inverse to reciprocal matrix, returning true if Matrix can be inverted. Geometrically, if Matrix maps from source to destination, inverse Matrix maps from destination to source. If Matrix can not be inverted, inverse is unchanged.

如果该CVMatrix可以求逆则对该CVMatrix的矩阵求逆，否则不变；几何意义上，如果该CVMatrix映射源到目标，则求逆后映射目标到源。

参数：
- `None`

返回：`None`

返回类型：`None`

---
### `write(value)`

Sets all values from input value. Sets matrix to: | sx skewX transX | | skewY sy transY | | persp0 persp1 persp2 |

将value值写入到CVMatrix中，value的值为float数组, 写入数据数量为 `min(len(value), 9)`

参数：
- `value:[float]` 写入值

返回：`None`

返回类型：`None`

---
### `read()`

Copies nine scalar values contained by Matrix into list, in member value ascending order: kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1, kMPersp2.

返回矩阵的9个值，以list的形式返回, 顺序为：kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1, kMPersp2

参数：
- `None`

返回：CVMatrix中的9个浮点值

返回类型：`list`

---
### `Example`
    
```python
import MNN
import MNN.cv as cv
import MNN.expr as expr
# CVMatrix创建
matrix = MNN.CVMatrix() # [[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]

# CVMatrix设置
matrix.setScale(2, 3, 4, 5) # [[2., 0., -4.], [0., 3., -10.], [-10., 0., 0.]]
matrix.preScale(-1, -3) # [[-2., 0., -4.], [0., -9., -10.], [-10., 0., 0.]]
matrix.setRotate(5) # [[0.996195, -0.087156, 0.0], [0.087156, 0.996195, 0.0], [0.0, 0.0, 1.0]]
matrix.setTranslate(5, 6) #[[1.0, 0.0, 5.0], [0.0, 1.0, 6.0], [0.0, 0.0, 1.0]]
matrix.setPolyToPoly([0, 0, 1, 1], [0, 1, 1, 0]) # [[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
matrix.invert() # [[0.0, -1.0, 1.0], [1.0, 0.0, -0.0], [0.0, 0.0, 1.0]]

# 常见图像处理用法
image = cv.imread('cat.jpg')
image_data = image.ptr
src_height, src_width, channel = image.shape
dst_height = dst_width = 224
dst_tensor = MNN.Tensor((1, dst_height, dst_width, channel), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Tensorflow)
# 1. 图像缩放
image_processer = MNN.CVImageProcess({'sourceFormat': MNN.CV_ImageFormat_BGR,
                                      'wrap': MNN.CV_Wrap_REPEAT,
                                      'destFormat': MNN.CV_ImageFormat_BGR})
height_scale = float(src_height / dst_height)
width_scale = float(src_width / dst_width)
matrix.setScale(width_scale, height_scale)
image_processer.setMatrix(matrix)
image_processer.convert(image_data, src_width, src_height, 0, dst_tensor)
scale_img = expr.const(dst_tensor.getHost(), [dst_height, dst_width, channel], expr.NHWC).astype(expr.uint8)
cv.imwrite('CVMatrix_scale.jpg', scale_img)
# 2. 图像填充
scale = max(height_scale, width_scale)
matrix.setScale(scale, scale)
resize_height = int(src_height / scale)
resize_width = int(src_width / scale)
if (dst_height - resize_height) > (dst_width - resize_width): # 从目标图片到源图片， 因此偏移应该用负值
    matrix.postTranslate(0, -(dst_height - resize_height) // 2 * scale)
else:
    matrix.postTranslate(-(dst_width - resize_width) // 2 * scale, 0)
image_processer.setMatrix(matrix)
image_processer.convert(image_data, src_width, src_height, 0, dst_tensor)
pad_img = expr.const(dst_tensor.getHost(), [dst_height, dst_width, channel], expr.NHWC).astype(expr.uint8)
cv.imwrite('CVMatrix_pad.jpg', pad_img)
# 3. 图像裁剪
image_processer = MNN.CVImageProcess({'sourceFormat': MNN.CV_ImageFormat_BGR,
                                      'wrap': MNN.CV_Wrap_ZERO,
                                      'destFormat': MNN.CV_ImageFormat_BGR})
matrix.setScale(width_scale, height_scale)
offset_y = (resize_height - dst_height) / 2 * height_scale
offset_x = (resize_width - dst_width) / 2 * width_scale
matrix.postTranslate(offset_x, offset_y)
image_processer.setMatrix(matrix)
image_processer.convert(image_data, src_width, src_height, 0, dst_tensor)
crop_img = expr.const(dst_tensor.getHost(), [dst_height, dst_width, channel], expr.NHWC).astype(expr.uint8)
cv.imwrite('CVMatrix_crop.jpg', crop_img)
# 4. 图像旋转
matrix.setScale(1 / src_width, 1 / src_height)
matrix.postRotate(30, 0.5, 0.5)
matrix.postScale(dst_width, dst_height)
matrix.invert() # 由于设置的是源图片到目标图片的变换矩阵， 因此取逆
image_processer.setMatrix(matrix)
image_processer.convert(image_data, src_width, src_height, 0, dst_tensor)
rotate_img = expr.const(dst_tensor.getHost(), [dst_height, dst_width, channel], expr.NHWC).astype(expr.uint8)
cv.imwrite('CVMatrix_rotate.jpg', rotate_img)
```

|    |    |
|:---|:---|
|![CVMatrix_scale.jpg](../_static/images/pymnn/CVMatrix_scale.jpg)|![CVMatrix_pad.jpg](../_static/images/pymnn/CVMatrix_pad.jpg)|
| CVMatrix_scale.jpg | CVMatrix_pad.jpg |
|![CVMatrix_crop.jpg](../_static/images/pymnn/CVMatrix_crop.jpg)|![CVMatrix_rotate.jpg](../_static/images/pymnn/CVMatrix_rotate.jpg)|
| CVMatrix_crop.jpg | CVMatrix_rotate.jpg |