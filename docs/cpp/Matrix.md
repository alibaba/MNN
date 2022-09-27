# Matrix
```cpp
class Matrix
```

## 枚举类
### TypeMask
```cpp
enum TypeMask {
    kIdentity_Mask    = 0,
    kTranslate_Mask   = 0x01,
    kScale_Mask       = 0x02,
    kAffine_Mask      = 0x04,
    kPerspective_Mask = 0x08
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `kIdentity_Mask` | 单位矩阵 |
| 1 | `kTranslate_Mask` | 转换矩阵 |
| 2 | `kScale_Mask` | 缩放矩阵 |
| 3 | `kAffine_Mask` | 倾斜或旋转矩阵 |
| 4 | `kPerspective_Mask` | 透视矩阵 |

---
### ScaleToFit
```cpp
enum ScaleToFit {
    kFill_ScaleToFit,
    kStart_ScaleToFit,
    kCenter_ScaleToFit,
    kEnd_ScaleToFit
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `kFill_ScaleToFit` | 缩放x和y来填充目标矩形 |
| 1 | `kStart_ScaleToFit` | 在左和上缩放和对齐 |
| 2 | `kCenter_ScaleToFit` | 中心缩放和对齐  |
| 3 | `kEnd_ScaleToFit` | 在右边和底部缩放和对齐 |

## 成员函数

---
### MakeScale
```cpp
static Matrix MakeScale(float sx, float sy) {
    Matrix m;
    m.setScale(sx, sy);
    return m;
};
```
设置矩阵缩放(sx, sy)，返回矩阵：| sx  0  0 |
                            |  0 sy  0 |
                            |  0  0  1 |

参数：
- `sx` 水平比例因子
- `sy` 垂直比例因子

返回：缩放矩阵

---
### MakeScale
```cpp
static Matrix MakeScale(float scale) {
    Matrix m;
    m.setScale(scale, scale);
    return m;
};
```
设置矩阵缩放(scale, scale)，返回矩阵：| scale  0    0 |
                                  |   0  scale  0 |
                                  |   0    0    1 |

参数：
- `scale` 水平比例因子

返回：缩放矩阵

---
### MakeTrans
```cpp
static Matrix MakeTrans(float dx, float dy) {
    Matrix m;
    m.setTranslate(dx, dy);
    return m;
};
```
设置矩阵平移到(dx, dy)，返回矩阵: | 1 0 dx |
                              | 0 1 dy |
                              | 0 0  1 |

参数：
- `dx` 水平平移
- `dy` 垂直平移

返回：平移矩阵

---
### MakeAll
```cpp
static Matrix MakeAll(float scaleX, float skewX, float transX, float skewY, float scaleY, float transY, float pers0,
                      float pers1, float pers2) {
    Matrix m;
    m.setAll(scaleX, skewX, transX, skewY, scaleY, transY, pers0, pers1, pers2);
    return m;
};
```
设置矩阵: | scaleX  skewX transX |
         |  skewY scaleY transY |
         |  pers0  pers1  pers2 |

参数：
- `scaleX` 水平比例因子
- `skewX` 水平倾斜因子
- `transX` 水平平移
- `skewY` 垂直倾斜因子
- `scaleY` 垂直比例因子
- `transY` 垂直平移
- `pers0` 输入x轴透视因子
- `pers1` 输入y轴透视因子
- `pers2` 透视比例因子

返回：矩阵

---
### getType
```cpp
TypeMask getType() const {
    if (fTypeMask & kUnknown_Mask) {
            fTypeMask = this->computeTypeMask();
    }
    return (TypeMask)(fTypeMask & 0xF);
};
```
返回一个位字段，描述矩阵可能进行的转换执行，位域是保守计算的。例如，当设置kPerspective_Mask时，all其他位被设置

参数：无

返回：kIdentity_Mask或kTranslate_Mask、kScale_Mask、kIdentity_Mask的组合kAffine_Mask, kPerspective_Mask

---
### isIdentity
```cpp
bool isIdentity() const {
    return this->getType() == 0;
};
```
如果矩阵是一致的则返回true，单位矩阵:| 1 0 0 |
                                | 0 1 0 |
                                | 0 0 1 |

参数：无

返回：如果矩阵是一致的则返回true

---
### isScaleTranslate
```cpp
bool isScaleTranslate() const {
    return !(this->getType() & ~(kScale_Mask | kTranslate_Mask));
};
```
矩阵可以是identity，只包含缩放元素，只包含平移元素，或同时包含二者。矩阵形式: | scale-x    0    translate-x |
                                                                   |    0    scale-y translate-y |
                                                                   |    0       0         1      |

参数：无

返回：如果矩阵是一致的，或者缩放，平移，或者两者兼而有之，则返回true

---
### isTranslate
```cpp
bool isTranslate() const {
    return !(this->getType() & ~(kTranslate_Mask));
};
```
矩阵形式: | 1 0 translate-x |
         | 0 1 translate-y |
         | 0 0      1      |

参数：无

返回：如果矩阵是一致的或者平移的，则返回true

---
### rectStaysRect
```cpp
bool rectStaysRect() const {
    if (fTypeMask & kUnknown_Mask) {
        fTypeMask = this->computeTypeMask();
    }
    return (fTypeMask & kRectStaysRect_Mask) != 0;
};
```
如果矩阵将一个矩形映射到另一个，则返回true，如果为true，矩阵是一致的，或缩放，或旋转90度的倍数，或者轴上的镜像。在所有情况下，矩阵也可以有平移。矩阵形式可以是：
            | scale-x    0    translate-x |
            |    0    scale-y translate-y |
            |    0       0         1      |

        or

            |    0     rotate-x translate-x |
            | rotate-y    0     translate-y |
            |    0        0          1      | 
对于非零的缩放-x，缩放-y，旋转-x和旋转-y，也称为preservesAxisAlignment()，使用提供更好内联文档的方法

参数：无

返回：如果矩阵将一个矩形映射到另一个，则返回true

---
### preservesAxisAlignment
```cpp
bool preservesAxisAlignment() const {
    return this->rectStaysRect();
};
```
矩阵将Rect映射到另一个Rect。如果为真，矩阵为恒等，或缩放，或旋转90度，或在轴上反射。在所有情况下，矩阵也可以有翻译。矩阵形式可以是：
            | scale-x    0    translate-x |
            |    0    scale-y translate-y |
            |    0       0         1      |

        or

            |    0     rotate-x translate-x |
            | rotate-y    0     translate-y |
            |    0        0          1      |
对于非零的缩放-x，缩放-y，旋转-x和旋转-y，也称为rectStaysRect()，使用提供更好内联文档的方法。

参数：无

返回：如果矩阵将一个矩形映射到另一个，则返回true

---
### operator
```cpp
float operator[](int index) const {
    MNN_ASSERT((unsigned)index < 9);
    return fMat[index];
};
```
返回一个矩阵值，如果索引超出范围并且定义了SK_DEBUG，则抛出

参数：
- `index` kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1, kMPersp2其中之一

返回：索引对应的值

---
### get
```cpp
float get(int index) const {
    MNN_ASSERT((unsigned)index < 9);
    return fMat[index];
};
```
返回一个矩阵值，如果索引超出范围并且定义了SK_DEBUG，则抛出

参数：
- `index` kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1, kMPersp2其中之一

返回：索引对应的值

---
### getScaleX
```cpp
float getScaleX() const {
    return fMat[kMScaleX];
};
```
返回比例因子 * x轴输入，影响x轴输出。通过mapPoints()方法，缩放点沿着x轴

参数：无

返回：水平缩放因子

---
### getScaleY
```cpp
float getScaleY() const {
    return fMat[kMScaleY];
};
```
返回比例因子 * y轴输入，影响y轴输出。通过mapPoints()方法，缩放点沿着y轴

参数：无

返回：垂直缩放因子

---
### getSkewY
```cpp
float getSkewY() const {
    return fMat[kMSkewY];
};
```
返回比例因子 * y轴输入，影响y轴输出。通过mapPoints()方法，沿着y轴倾斜角度。倾斜两个轴可以旋转角度

参数：无

返回：垂直倾斜因子

---
### getSkewX
```cpp
float getSkewX() const {
    return fMat[kMSkewX];
};
```
返回比例因子 * x轴输入，影响x轴输出。通过mapPoints()方法，沿着x轴倾斜角度。倾斜两个轴可以旋转角度

参数：无

返回：水平倾斜因子

---
### getTranslateX
```cpp
float getTranslateX() const {
    return fMat[kMTransX];
};
```
返回用于x轴输出的平移。通过mapPoints()方法，沿着x轴移动

参数：无

返回：水平移动因子

---
### getTranslateY
```cpp
float getTranslateY() const {
    return fMat[kMTransY];
};
```
返回用于y轴输出的平移。通过mapPoints()方法，沿着y轴移动

参数：无

返回：垂直移动因子

---
### getPerspX
```cpp
float getPerspX() const {
    return fMat[kMPersp0];
};
```
返回x轴缩放输入相对于y轴缩放输入的缩放因子

参数：无

返回：x轴输入的角度因子

---
### getPerspY
```cpp
float getPerspY() const {
    return fMat[kMPersp1];
};
```
返回y轴缩放输入相对于x轴缩放输入的缩放因子

参数：无

返回：y轴输入的角度因子

---
### operator
```cpp
float& operator[](int index) {
    MNN_ASSERT((unsigned)index < 9);
    this->setTypeMask(kUnknown_Mask);
    return fMat[index];
};
```
返回可写的矩阵值，如果索引超出范围并且定义了SK_DEBUG，则抛出。清除内部缓存，预计调用者将更改矩阵值。下一次读取矩阵状态可能会重新计算缓存，随后对矩阵值的写入必须在dirtyMatrixTypeCache()之后。

参数：
- `index` kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1, kMPersp2其中之一

返回：索引对应的可写值

---
### set
```cpp
void set(int index, float value) {
    MNN_ASSERT((unsigned)index < 9);
    fMat[index] = value;
    this->setTypeMask(kUnknown_Mask);
};
```
返回矩阵值，如果索引超出范围并且定义了SK_DEBUG，则抛出。比运营商安全，始终维护内部缓存

参数：
- `index` kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1, kMPersp2其中之一
- `value` 存储在矩阵中的标量

返回：`void`

---
### setScaleX
```cpp
void setScaleX(float v) {
    this->set(kMScaleX, v);
};
```
设置水平比例因子

参数：
- `v` 存储的水平比例因子

返回：`void`

---
### setScaleY
```cpp
void setScaleY(float v) {
    this->set(kMScaleY, v);
};
```
设置垂直比例因子

参数：
- `v` 存储的垂直比例因子

返回：`void`

---
### setSkewY
```cpp
void setSkewY(float v) {
    this->set(kMSkewY, v);
};
```
设置垂直倾斜因子

参数：
- `v` 存储的垂直倾斜因子

返回：`void`

---
### setSkewX
```cpp
void setSkewX(float v) {
    this->set(kMSkewX, v);
};
```
设置水平倾斜因子

参数：
- `v` 存储的水平倾斜因子

返回：`void`

---
### setTranslateX
```cpp
void setTranslateX(float v) {
    this->set(kMTransX, v);
};
```
设置水平平移因子

参数：
- `v` 存储的水平平移因子

返回：`void`

---
### setTranslateY
```cpp
void setTranslateY(float v) {
    this->set(kMTransY, v);
};
```
设置垂直平移因子

参数：
- `v` 存储的垂直平移因子

返回：`void`

---
### setPerspX
```cpp
void setPerspX(float v) {
    this->set(kMPersp0, v);
};
```
设置输入x轴透视因子，它会导致mapXY()改变输入x轴值与输入y轴值成反比

参数：
- `v` 存储的x轴透视因子

返回：`void`

---
### setPerspY
```cpp
void setPerspY(float v) {
    this->set(kMPersp1, v);
};
```
设置输入y轴透视因子，它会导致mapXY()以输入y轴值与输入x轴值成反比的方式改变输入y轴值

参数：
- `v` 存储的y轴透视因子

返回：`void`

---
### setAll
```cpp
void setAll(float scaleX, float skewX, float transX, float skewY, float scaleY, float transY, float persp0,
            float persp1, float persp2) {
    fMat[kMScaleX] = scaleX;
    fMat[kMSkewX]  = skewX;
    fMat[kMTransX] = transX;
    fMat[kMSkewY]  = skewY;
    fMat[kMScaleY] = scaleY;
    fMat[kMTransY] = transY;
    fMat[kMPersp0] = persp0;
    fMat[kMPersp1] = persp1;
    fMat[kMPersp2] = persp2;
    this->setTypeMask(kUnknown_Mask);
};
```
根据参数设置所有值，设置矩阵: | scaleX  skewX transX |
                         |  skewY scaleY transY |
                         |  pers0  pers1  pers2 |

参数：
- `scaleX` 存储的水平比例因子
- `skewX` 存储的水平倾斜因子
- `transX` 存储的水平平移因子
- `skewY` 存储的垂直倾斜因子
- `scaleY` 存储的垂直比例因子
- `transY` 存储的垂直平移因子
- `pers0` 存储的输入x轴透视因子
- `pers1` 存储的输入y轴透视因子
- `pers2` 存储的透视比例因子

返回：矩阵

---
### get9
```cpp
void get9(float buffer[9]) const {
    memcpy(buffer, fMat, 9 * sizeof(float));
};
```
将矩阵中包含的9个标量值按成员值升序复制到缓冲区:kMScaleX、kMSkewX、kMTransX、kMSkewY、kMScaleY、kMTransY、kMPersp0、kMPersp1、kMPersp2

参数：
- `buffer[9]` 存储九个标量值

返回：`void`

---
### set9
```cpp
void set9(const float buffer[9]);
```
设置矩阵缓冲区中的9个标量值，成员值按升序排列:
kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1, kMPersp2
设置矩阵：
            | buffer[0] buffer[1] buffer[2] |
            | buffer[3] buffer[4] buffer[5] |
            | buffer[6] buffer[7] buffer[8] |
将来，set9后跟get9可能不会返回相同的值。由于矩阵映射非齐次坐标，缩放所有9个值产生了等效变换，可能会提高精度

参数：
- `buffer[9]` 九个标量值

返回：`void`

---
### reset
```cpp
void reset();
```
设置矩阵单位，这对映射的点没有影响。设置矩阵：| 1 0 0 |
                                      | 0 1 0 |
                                      | 0 0 1 |
也称为setIdentity()，使用提供更好内联的那个文档

参数：无

返回：`void`

---
### setIdentity
```cpp
void setIdentity() {
    this->reset();
};
```
设置矩阵单位，这对映射的点没有影响。设置矩阵：| 1 0 0 |
                                      | 0 1 0 |
                                      | 0 0 1 |
也称为reset()，使用提供更好内联的那个文档

参数：无

返回：`void`

---
### setTranslate
```cpp
void setTranslate(float dx, float dy);
```
设置矩阵平移到(dx, dy)

参数：
- `dx` 水平平移
- `dy` 垂直平移

返回：`void`

---
### setScale
```cpp
void setScale(float sx, float sy, float px, float py);
```
设置矩阵缩放sx和sy，大约一个枢轴点(px, py)，当映射到矩阵时，枢轴点是不变的

参数：
- `sx` 水平缩放因子
- `sy` 垂直缩放因子
- `px` x轴
- `py` y轴

返回：`void`

---
### setScale
```cpp
void setScale(float sx, float sy);
```
设置矩阵在(0,0)的枢轴点处按sx和sy缩放

参数：
- `sx` 水平缩放因子
- `sy` 垂直缩放因子

返回：`void`

---
### setRotate
```cpp
void setRotate(float degrees, float px, float py);
```
设置矩阵以轴点(px, py)旋转角度，当映射到矩阵时，枢轴点是不变的，正度顺时针旋转

参数：
- `degrees` 水平坐标轴与垂直坐标轴的夹角
- `sx` 水平缩放因子
- `sy` 垂直缩放因子

返回：`void`

---
### setSinCos
```cpp
void setSinCos(float sinValue, float cosValue, float px, float py);
```
设置矩阵旋转sinValue和cosValue，旋转一个轴心点(px, py)。当映射到矩阵时，轴点是不变的，向量(sinValue, cosValue)描述相对于(0,1)的旋转角度，向量长度指定缩放

参数：
- `sinValue` 旋转向量x轴部分
- `cosValue` 旋转向量y轴部分
- `sx` 水平缩放因子
- `sy` 垂直缩放因子

返回：`void`

---
### setSinCos
```cpp
void setSinCos(float sinValue, float cosValue);
```
设置矩阵的sinValue和cosValue旋转，大约在(0,0)的轴点。向量(sinValue, cosValue)描述相对于(0,1)的旋转角度，向量长度指定缩放

参数：
- `sinValue` 旋转向量x轴部分
- `cosValue` 旋转向量y轴部分

返回：`void`

---
### setSkew
```cpp
void setSkew(float kx, float ky, float px, float py);
```
设置矩阵在kx和ky上的倾斜，关于一个轴点(px, py)，当映射到矩阵时，轴点是不变的

参数：
- `kx` 水平倾斜因子
- `ky` 垂直倾斜因子
- `px` x轴
- `py` y轴

返回：`void`

---
### setConcat
```cpp
void setConcat(const Matrix& a, const Matrix& b);
```
将矩阵设为矩阵a乘以矩阵b，a或b都可以是这个
假定：
                | A B C |      | J K L |
            a = | D E F |, b = | M N O |
                | G H I |      | P Q R |
设置矩阵：
                    | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
            a * b = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
                    | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |

参数：
- `a` 乘法表达式的左边矩阵
- `b` 乘法表达式的右边矩阵

返回：`void`

---
### preTranslate
```cpp
void preTranslate(float dx, float dy);
```
设置矩阵到矩阵乘以由平移(dx, dy)构造的矩阵，这可以被认为是在应用矩阵之前移动要映射的点
假定：
                     | A B C |               | 1 0 dx |
            Matrix = | D E F |,  T(dx, dy) = | 0 1 dy |
                     | G H I |               | 0 0  1 |
设置矩阵：
                                 | A B C | | 1 0 dx |   | A B A*dx+B*dy+C |
            Matrix * T(dx, dy) = | D E F | | 0 1 dy | = | D E D*dx+E*dy+F |
                                 | G H I | | 0 0  1 |   | G H G*dx+H*dy+I |

参数：
- `dx` 应用矩阵前在x轴平移
- `dy` 应用矩阵前在y轴平移

返回：`void`

---
### preScale
```cpp
void preScale(float sx, float sy, float px, float py);
```
在应用矩阵之前缩放一个轴点
假定：
                     | A B C |               | 1 0 dx |
            Matrix = | D E F |,  T(dx, dy) = | 0 1 dy |
                     | G H I |               | 0 0  1 |
目标：

            dx = px - sx * px
            dy = py - sy * py
设置矩阵：
                                 | A B C | | 1 0 dx |   | A B A*dx+B*dy+C |
            Matrix * T(dx, dy) = | D E F | | 0 1 dy | = | D E D*dx+E*dy+F |
                                 | G H I | | 0 0  1 |   | G H G*dx+H*dy+I |

参数：
- `sx` 水平比例因子
- `sy` 垂直比例因子
- `px` x轴
- `py` y轴

返回：`void`

---
### preScale
```cpp
void preScale(float sx, float sy);
```
在应用矩阵之前缩放原点
假定：
                     | A B C |               | sx  0  0 |
            Matrix = | D E F |,  S(sx, sy) = |  0 sy  0 |
                     | G H I |               |  0  0  1 |
目标：
            c  = cos(degrees)
            s  = sin(degrees)
            dx =  s * py + (1 - c) * px
            dy = -s * px + (1 - c) * py

设置矩阵：
                                          | A B C | | c -s dx |   | Ac+Bs -As+Bc A*dx+B*dy+C |
            Matrix * R(degrees, px, py) = | D E F | | s  c dy | = | Dc+Es -Ds+Ec D*dx+E*dy+F |
                                          | G H I | | 0  0  1 |   | Gc+Hs -Gs+Hc G*dx+H*dy+I |
参数：
- `sx` 水平比例因子
- `sy` 垂直比例因子

返回：`void`

---
### preRotate
```cpp
void preRotate(float degrees, float px, float py);
```
在应用矩阵之前绕一个轴点旋转，顺时针旋转为正
假定：
                     | A B C |                        | c -s dx |
            Matrix = | D E F |,  R(degrees, px, py) = | s  c dy |
                     | G H I |                        | 0  0  1 |
目标：
            c  = cos(degrees)
            s  = sin(degrees)
            dx =  s * py + (1 - c) * px
            dy = -s * px + (1 - c) * py
设置矩阵：

                                          | A B C | | c -s dx |   | Ac+Bs -As+Bc A*dx+B*dy+C |
            Matrix * R(degrees, px, py) = | D E F | | s  c dy | = | Dc+Es -Ds+Ec D*dx+E*dy+F |
                                          | G H I | | 0  0  1 |   | Gc+Hs -Gs+Hc G*dx+H*dy+I |

参数：
- `degrees` 坐标轴与垂直坐标轴的夹角
- `px` x轴
- `py` y轴

返回：`void`

---
### preRotate
```cpp
void preRotate(float degrees);
```
应用矩阵之前绕原点旋转，顺时针旋转为正
假定：
                     | A B C |                        | c -s dx |
            Matrix = | D E F |,  R(degrees, px, py) = | s  c dy |
                     | G H I |                        | 0  0  1 |
目标：
            c  = cos(degrees)
            s  = sin(degrees)
设置矩阵：
                                          | A B C | | c -s 0 |   | Ac+Bs -As+Bc C |
            Matrix * R(degrees, px, py) = | D E F | | s  c 0 | = | Dc+Es -Ds+Ec F |
                                          | G H I | | 0  0 1 |   | Gc+Hs -Gs+Hc I |

参数：
- `degrees` 坐标轴与垂直坐标轴的夹角

返回：`void`

---
### preSkew
```cpp
void preSkew(float kx, float ky, float px, float py);
```
应用矩阵之前绕一个轴点倾斜
假定：
                     | A B C |                       |  1 kx dx |
            Matrix = | D E F |,  K(kx, ky, px, py) = | ky  1 dy |
                     | G H I |                       |  0  0  1 |
目标：
            dx = -kx * py
            dy = -ky * px
设置矩阵：
                                         | A B C | |  1 kx dx |   | A+B*ky A*kx+B A*dx+B*dy+C |
            Matrix * K(kx, ky, px, py) = | D E F | | ky  1 dy | = | D+E*ky D*kx+E D*dx+E*dy+F |
                                         | G H I | |  0  0  1 |   | G+H*ky G*kx+H G*dx+H*dy+I |

参数：
- `kx` 水平倾斜因子
- `ky` 垂直倾斜因子
- `px` x轴
- `py` y轴

返回：`void`

---
### preSkew
```cpp
void preSkew(float kx, float ky);
```
应用矩阵之前绕原点倾斜
假定：
                     | A B C |               |  1 kx 0 |
            Matrix = | D E F |,  K(kx, ky) = | ky  1 0 |
                     | G H I |               |  0  0 1 |
设置矩阵：
                                 | A B C | |  1 kx 0 |   | A+B*ky A*kx+B C |
            Matrix * K(kx, ky) = | D E F | | ky  1 0 | = | D+E*ky D*kx+E F |
                                 | G H I | |  0  0 1 |   | G+H*ky G*kx+H I |

参数：
- `kx` 水平倾斜因子
- `ky` 垂直倾斜因子

返回：`void`

---
### preConcat
```cpp
void preConcat(const Matrix& other);
```
在应用矩阵之前的映射
假定：
                     | A B C |          | J K L |
            Matrix = | D E F |, other = | M N O |
                     | G H I |          | P Q R |
设置矩阵：
                             | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
            Matrix * other = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
                             | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |

参数：
- `other` 乘法表达式的右边矩阵

返回：`void`

---
### postTranslate
```cpp
void postTranslate(float dx, float dy);
```
应用矩阵后移动被映射的点
假定：
                     | J K L |               | 1 0 dx |
            Matrix = | M N O |,  T(dx, dy) = | 0 1 dy |
                     | P Q R |               | 0 0  1 |
设置矩阵：
                                 | 1 0 dx | | J K L |   | J+dx*P K+dx*Q L+dx*R |
            T(dx, dy) * Matrix = | 0 1 dy | | M N O | = | M+dy*P N+dy*Q O+dy*R |
                                 | 0 0  1 | | P Q R |   |      P      Q      R |

参数：
- `dx` 应用矩阵后的x轴平移
- `dy` 应用矩阵后的y轴平移

返回：`void`

---
### postScale
```cpp
void postScale(float sx, float sy, float px, float py);
```
应用矩阵后缩放一个轴点
假定：
                     | J K L |                       | sx  0 dx |
            Matrix = | M N O |,  S(sx, sy, px, py) = |  0 sy dy |
                     | P Q R |                       |  0  0  1 |
目标：
            dx = px - sx * px
            dy = py - sy * py
设置矩阵：
                                         | sx  0 dx | | J K L |   | sx*J+dx*P sx*K+dx*Q sx*L+dx+R |
            S(sx, sy, px, py) * Matrix = |  0 sy dy | | M N O | = | sy*M+dy*P sy*N+dy*Q sy*O+dy*R |
                                         |  0  0  1 | | P Q R |   |         P         Q         R |
参数：
- `sx` 水平比例因子
- `sy` 垂直比例因子
- `px` x轴
- `py` y轴

返回：`void`

---
### postScale
```cpp
void postScale(float sx, float sy);
```
应用矩阵之后关于原点的缩放
假定：
                     | J K L |               | sx  0  0 |
            Matrix = | M N O |,  S(sx, sy) = |  0 sy  0 |
                     | P Q R |               |  0  0  1 |
设置矩阵：
                                 | sx  0  0 | | J K L |   | sx*J sx*K sx*L |
            S(sx, sy) * Matrix = |  0 sy  0 | | M N O | = | sy*M sy*N sy*O |
                                 |  0  0  1 | | P Q R |   |    P    Q    R |
参数：
- `sx` 水平比例因子
- `sy` 垂直比例因子

返回：`void`

---
### postIDiv
```cpp
bool postIDiv(int divx, int divy);
```
应用矩阵之后按照(1/divx, 1/divy)比例缩放一个枢轴点
假定：
                     | J K L |                   | sx  0  0 |
            Matrix = | M N O |,  I(divx, divy) = |  0 sy  0 |
                     | P Q R |                   |  0  0  1 |
目标：
            sx = 1 / divx
            sy = 1 / divy
设置矩阵：
                                     | sx  0  0 | | J K L |   | sx*J sx*K sx*L |
            I(divx, divy) * Matrix = |  0 sy  0 | | M N O | = | sy*M sy*N sy*O |
                                     |  0  0  1 | | P Q R |   |    P    Q    R |
参数：
- `divx` x逆比例的整数除数
- `divy` y逆比例的整数除数

返回：缩放成功返回true

---
### postRotate
```cpp
void postRotate(float degrees, float px, float py);
```
应用矩阵后绕一个枢轴点旋转
假定：
                     | J K L |                        | c -s dx |
            Matrix = | M N O |,  R(degrees, px, py) = | s  c dy |
                     | P Q R |                        | 0  0  1 |
目标：
            c  = cos(degrees)
            s  = sin(degrees)
            dx =  s * py + (1 - c) * px
            dy = -s * px + (1 - c) * py
设置矩阵：
                                          |c -s dx| |J K L|   |cJ-sM+dx*P cK-sN+dx*Q cL-sO+dx+R|
            R(degrees, px, py) * Matrix = |s  c dy| |M N O| = |sJ+cM+dy*P sK+cN+dy*Q sL+cO+dy*R|
                                          |0  0  1| |P Q R|   |         P          Q          R|
参数：
- `degrees` 坐标轴与垂直坐标轴的夹角
- `px` x轴
- `py` y轴

返回：`void`

---
### postRotate
```cpp
void postRotate(float degrees);
```
应用矩阵后绕原点旋转
假定：
                     | J K L |                        | c -s 0 |
            Matrix = | M N O |,  R(degrees, px, py) = | s  c 0 |
                     | P Q R |                        | 0  0 1 |
目标：
            c  = cos(degrees)
            s  = sin(degrees)
设置矩阵：
                                          | c -s dx | | J K L |   | cJ-sM cK-sN cL-sO |
            R(degrees, px, py) * Matrix = | s  c dy | | M N O | = | sJ+cM sK+cN sL+cO |
                                          | 0  0  1 | | P Q R |   |     P     Q     R |
参数：
- `degrees` 坐标轴与垂直坐标轴的夹角

返回：`void`

---
### postSkew
```cpp
void postSkew(float kx, float ky, float px, float py);
```
应用矩阵后绕一个枢轴点倾斜
假定：
                     | J K L |                       |  1 kx dx |
            Matrix = | M N O |,  K(kx, ky, px, py) = | ky  1 dy |
                     | P Q R |                       |  0  0  1 |
目标：
            dx = -kx * py
            dy = -ky * px
设置矩阵：
                                         | 1 kx dx| |J K L|   |J+kx*M+dx*P K+kx*N+dx*Q L+kx*O+dx+R|
            K(kx, ky, px, py) * Matrix = |ky  1 dy| |M N O| = |ky*J+M+dy*P ky*K+N+dy*Q ky*L+O+dy*R|
                                         | 0  0  1| |P Q R|   |          P           Q           R|
参数：
- `kx` 水平倾斜因子
- `ky` 垂直倾斜因子
- `px` x轴
- `py` y轴

返回：`void`

---
### postSkew
```cpp
void postSkew(float kx, float ky);
```
应用矩阵后绕一个枢轴点倾斜
假定：
                     | J K L |               |  1 kx 0 |
            Matrix = | M N O |,  K(kx, ky) = | ky  1 0 |
                     | P Q R |               |  0  0 1 |
设置矩阵：
                                 |  1 kx 0 | | J K L |   | J+kx*M K+kx*N L+kx*O |
            K(kx, ky) * Matrix = | ky  1 0 | | M N O | = | ky*J+M ky*K+N ky*L+O |
                                 |  0  0 1 | | P Q R |   |      P      Q      R |
参数：
- `kx` 水平倾斜因子
- `ky` 垂直倾斜因子

返回：`void`

---
### postConcat
```cpp
void postConcat(const Matrix& other);
```
设置矩阵到矩阵其他乘以矩阵，这可以被认为是映射后，其他应用矩阵
假定：
                     | J K L |           | A B C |
            Matrix = | M N O |,  other = | D E F |
                     | P Q R |           | G H I |
设置矩阵：
                             | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
            other * Matrix = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
                             | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |
参数：
- `other` 乘法表达式的左边矩阵

返回：`void`

---
### setRectToRect
```cpp
bool setRectToRect(const Rect& src, const Rect& dst, ScaleToFit stf);
```
设置矩阵缩放并将src Rect转换为dst，recf选择映射是否完全填充dst或保留长宽比，以及如何在dst内对齐src。如果src为空则返回false，并设置矩阵为identity。如果dst为空则返回true
设置矩阵：| 0 0 0 |
        | 0 0 0 |
        | 0 0 1 |
参数：
- `src` 要映射的rect
- `dst` 要映射到的rect
- `stf` kFill_ScaleToFit, kStart_ScaleToFit,kCenter_ScaleToFit, kEnd_ScaleToFit其中之一

返回：如果矩阵可以表示Rect映射，则为true

---
### MakeRectToRect
```cpp
static Matrix MakeRectToRect(const Rect& src, const Rect& dst, ScaleToFit stf) {
    Matrix m;
    m.setRectToRect(src, dst, stf);
    return m;
};
```
返回矩阵设置为缩放并将src Rect转换为dst，recf选择映射是否完全填充dst或保留长宽比，以及如何在dst内对齐src。如果src为空，则返回标识矩阵。如果dst为空，返回设置矩阵：| 0 0 0 |
                     | 0 0 0 |
                     | 0 0 1 |
参数：
- `src` 要映射的rect
- `dst` 要映射到的rect
- `stf` kFill_ScaleToFit, kStart_ScaleToFit,kCenter_ScaleToFit, kEnd_ScaleToFit其中之一

返回：将src映射到dst的矩阵

---
### setPolyToPoly
```cpp
bool setPolyToPoly(const Point src[], const Point dst[], int count);
```
设置“矩阵”将src映射到dst，Count必须为0或更大，4或更小。
如果count为零，设置Matrix为identity并返回true。
如果count为1，设置Matrix转换并返回true。
如果count是两个或更多，设置矩阵映射点，如果可能;返回false
如果矩阵不能被构造。如果计数是4，矩阵可能包括透视。

参数：
- `src[]` 要映射的rect
- `dst[]` 要映射到的rect
- `count` 在scr和dst中点的数量

返回：如果矩阵构造成功，返回true

---
### invert
```cpp
bool invert(Matrix* inverse) const {
    if (this->isIdentity()) {
        if (inverse) {
            inverse->reset();
        }
        return true;
    }
    return this->invertNonIdentity(inverse);
};
```
矩阵反转，几何上，如果矩阵从源映射到目标，则逆矩阵从目标映射到源。如果矩阵不能被反转，逆矩阵不变

参数：
- `inverse` 要被反转的矩阵，可能是nullptr

返回：矩阵反转成功，返回true

---
### SetAffineIdentity
```cpp
static void SetAffineIdentity(float affine[6]);
```
在主序列中用标识值填充仿射
设置仿射：
            | 1 0 0 |
            | 0 1 0 |
OpenGL和XPS在主序列中仿射3x2矩阵

参数：
- `affine` 3x2仿射矩阵

返回：`void`

---
### asAffine
```cpp
bool asAffine(float affine[6]) const;
```
在主序列中填充仿射
设置仿射：
            | scale-x  skew-x translate-x |
            | skew-y  scale-y translate-y |
如果矩阵包含透视图，则返回false并保持仿射不变

参数：
- `affine` 3x2仿射矩阵，可能是nullptr

返回：如果矩阵不包含透视图，则返回true

---
### setAffine
```cpp
void setAffine(const float affine[6]);
```
将矩阵设置为仿射值，按主序列传递，给定仿射，列|行
例如：
            | scale-x  skew-x translate-x |
            |  skew-y scale-y translate-y |

矩阵是集合，行|列
例如：
            | scale-x  skew-x translate-x |
            |  skew-y scale-y translate-y |
            |       0       0           1 |

参数：
- `affine` 3 x2仿射矩阵

返回：`void`

---
### mapPoints
```cpp
void mapPoints(Point dst[], const Point src[], int count) const {
    MNN_ASSERT((dst && src && count > 0) || 0 == count);
    MNN_ASSERT(src == dst || &dst[count] <= &src[0] || &src[count] <= &dst[0]);
    this->getMapPtsProc()(*this, dst, src, count);
};
```
映射指定长度计数的点数组，通过将每个点乘以矩阵来映射点
假定：
                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |
目标：
            for (i = 0; i < count; ++i) {
                x = pts[i].fX
                y = pts[i].fY
            }
每一个点的计算结果为：
                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I
SRC和DST可能指向相同的存储空间

参数：
- `dst` 映射点存储空间
- `src` 变换点
- `count` 变换点的个数

返回：`void`

---
### mapPoints
```cpp
void mapPoints(Point pts[], int count) const {
    this->mapPoints(pts, pts, count);
};
```
映射指定长度计数的点数组，通过将每个点乘以矩阵来映射点
假定：
                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |
目标：
            for (i = 0; i < count; ++i) {
                x = pts[i].fX
                y = pts[i].fY
            }
每一个点的计算结果为：
                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I
SRC和DST可能指向相同的存储空间

参数：
- `pts` 映射点存储空间
- `count` 变换点的个数

返回：`void`

---
### mapXY
```cpp
void mapXY(float x, float y, Point* result) const {
    this->getMapXYProc()(*this, x, y, result);
};
```
点(x, y)的映射结果，点通过乘以矩阵来映射
假定：
                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |
计算结果为：
                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

参数：
- `x` 要映射的点的x轴值
- `y` 要映射的点的y轴值
- `result` 映射点的存储

返回：`void`

---
### mapXY
```cpp
Point mapXY(float x, float y) const {
    Point result;
    this->getMapXYProc()(*this, x, y, &result);
    return result;
};
```
点(x, y)的映射结果，点通过乘以矩阵来映射
假定：
                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |
计算结果为：
                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

参数：
- `x` 要映射的点的x轴值
- `y` 要映射的点的y轴值

返回：映射点

---
### mapRect
```cpp
bool mapRect(Rect* dst, const Rect& src) const;
```
将dst设置为矩阵映射的src角的边界，如果映射的角是dst角则返回true。返回值与调用rectStaysRect()方法相同

参数：
- `dst` 存储的映射点的边界
- `src` 要绘制的rect

返回：如果DST等价于映射的SRC，则为True

---
### mapRect
```cpp
bool mapRect(Rect* rect) const {
    return this->mapRect(rect, *rect);
};
```
将rect设置为矩阵映射的矩形角的边界，如果映射的角是计算出来的矩形角，则返回true，返回值与调用rectStaysRect()相同

参数：
- `rect` 要映射的rect，并存储映射角的边界

返回：如果结果等价于映射的SRC，则为True

---
### mapRect
```cpp
Rect mapRect(const Rect& src) const {
    Rect dst;
    (void)this->mapRect(&dst, src);
    return dst;
};
```
返回由矩阵映射的src角的边界

参数：
- `src` 要绘制的矩形

返回：映射的边界

---
### mapRectScaleTranslate
```cpp
void mapRectScaleTranslate(Rect* dst, const Rect& src) const;
```
将dst设置为矩阵映射的src角的边界，如果矩阵包含了缩放或转换以外的元素:如果SK_DEBUG被定义了则生效，否则结果为undefined

参数：
- `dst` 存储映射点的边界
- `src` 要绘制的Rect

返回：`void`

---
### cheapEqualTo
```cpp
bool cheapEqualTo(const Matrix& m) const {
    return 0 == memcmp(fMat, m.fMat, sizeof(fMat));
};
```
如果矩阵等于m，则返回true；当zero值的符号不同时返回false；当一个矩阵为正zero另一个矩阵为负zero时，即使两个矩阵都包含NaN，也返回true。NaN从不等于任何值，包括它自己。为了提高性能，如果NaN值的位模式相等，则将其视为相等的位模式。

参数：
- `m` 被比较的矩阵

返回：如果m和矩阵由相同的位模式表示，则为true

---
### operator==
```cpp
friend MNN_PUBLIC bool operator==(const Matrix& a, const Matrix& b);
```
比较a和b，如果a和b在数值上相等，返回true。即使zero值的符号不同，也返回true。如果其中一个矩阵包含NaN，则返回false，即使另一个矩阵也包含NaN

参数：
- `a` 被比较的矩阵a
- `b` 被比较的矩阵b

返回：当矩阵a和矩阵b在数值上相等时为true

---
### operator!=
```cpp
friend MNN_PUBLIC bool operator!=(const Matrix& a, const Matrix& b) {
    return !(a == b);
};
```
比较a和b，如果a和b在数值上不相等，则返回true。即使zero值的符号不同，也返回false。如果其中一个矩阵包含NaN，则返回true，即使另一个矩阵也包含NaN

参数：
- `a` 被比较的矩阵a
- `b` 被比较的矩阵b

返回：如果矩阵a和矩阵b在数值上不相等，则为true

---
### dump
```cpp
void dump() const;
```
将矩阵的文本表示形式写入标准输出，浮点值的写入精度有限，可能无法重建原始矩阵的输出

参数：无

返回：`void`

---
### getMinScale
```cpp
float getMinScale() const;
```
通过分解缩放和倾斜元素，返回矩阵的最小缩放因子。如果比例因子溢出或矩阵包含透视图，则返回-1

参数：无

返回：最小缩放因子

---
### getMaxScale
```cpp
float getMaxScale() const;
```
通过分解缩放和倾斜元素，返回矩阵的最大缩放因子。如果比例因子溢出或矩阵包含透视图，则返回-1

参数：无

返回：最大缩放因子

---
### getMinMaxScales
```cpp
bool getMinMaxScales(float scaleFactors[2]) const;
```
将scaleFactors[0]设置为最小缩放因子，将scaleFactors[1]设置为最大缩放因子。缩放因子是通过分解矩阵缩放和倾斜元素来计算的。如果找到scaleFactors则返回true，否则，返回false，并将scaleFactors设置为未定义的值

参数：
- `scaleFactors` 最小和最大的缩放因子

返回：如果缩放因子计算正确，则返回true

---
### I
```cpp
static const Matrix& I();
```
返回对单位矩阵常量的引用，返回矩阵被设置为：
            | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |

参数：无

返回：单位矩阵常量

---
### InvalidMatrix
```cpp
static const Matrix& InvalidMatrix();
```
返回指向一个值无效的常量矩阵的引用，返回矩阵被设置为：
            | SK_ScalarMax SK_ScalarMax SK_ScalarMax |
            | SK_ScalarMax SK_ScalarMax SK_ScalarMax |
            | SK_ScalarMax SK_ScalarMax SK_ScalarMax |

参数：无

返回：无效的常量矩阵

---
### Concat
```cpp
static Matrix Concat(const Matrix& a, const Matrix& b) {
    Matrix result;
    result.setConcat(a, b);
    return result;
};
```
返回矩阵a乘以矩阵b
假定：
                | A B C |      | J K L |
            a = | D E F |, b = | M N O |
                | G H I |      | P Q R |
设置矩阵为：
                    | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
            a * b = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
                    | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |

参数：
- `a` 乘法表达式的左边矩阵
- `b` 乘法表达式的右边矩阵

返回：无效的常量矩阵

---
### dirtyMatrixTypeCache
```cpp
void dirtyMatrixTypeCache() {
    this->setTypeMask(kUnknown_Mask);
};
```
将内部缓存设置为未知状态，用于在对操作符[](int index)返回的矩阵元素引用进行重复修改后强制更新

参数：无

返回：`void`

---
### setScaleTranslate
```cpp
void setScaleTranslate(float sx, float sy, float tx, float ty) {
    fMat[kMScaleX] = sx;
    fMat[kMSkewX]  = 0;
    fMat[kMTransX] = tx;

    fMat[kMSkewY]  = 0;
    fMat[kMScaleY] = sy;
    fMat[kMTransY] = ty;

    fMat[kMPersp0] = 0;
    fMat[kMPersp1] = 0;
    fMat[kMPersp2] = 1;

    unsigned mask = 0;
    if (sx != 1 || sy != 1) {
        mask |= kScale_Mask;
    }
    if (tx || ty) {
        mask |= kTranslate_Mask;
    }
    this->setTypeMask(mask | kRectStaysRect_Mask);
};
```
使用缩放和转换元素初始化矩阵
            | sx  0 tx |
            |  0 sy ty |
            |  0  0  1 |

参数：
- `sx` 水平缩放因子
- `sy` 垂直缩放因子
- `tx` 水平平移因子
- `ty` 垂直平移因子

返回：`void`