# ImageProcess
```cpp
class ImageProcess
```

## 枚举类
### ImageFormat
```cpp
enum ImageFormat {
    RGBA     = 0,
    RGB      = 1,
    BGR      = 2,
    GRAY     = 3,
    BGRA     = 4,
    YCrCb    = 5,
    YUV      = 6,
    HSV      = 7,
    XYZ      = 8,
    BGR555   = 9,
    BGR565   = 10,
    YUV_NV21 = 11,
    YUV_NV12 = 12,
    YUV_I420 = 13,
    HSV_FULL = 14,
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 |  `RGBA`     |  |
| 1 |  `RGB`      |  |
| 2 |  `BGR`      |  |
| 3 |  `GRAY`     |  |
| 4 |  `BGRA`     |  |
| 5 |  `YCrCb`    |  |
| 6 |  `YUV`      |  |
| 7 |  `HSV`      |  |
| 8 |  `XYZ`      |  |
| 9 |  `BGR555`   |  |
| 10 | `BGR565`   |  |
| 11 | `YUV_NV21` |  |
| 12 | `YUV_NV12` |  |
| 13 | `YUV_I420` |  |
| 14 | `HSV_FULL` |  |


---
### Filter
```cpp
enum Filter {
    NEAREST       = 0,
    BILINEAR      = 1,
    BICUBIC       = 2,
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `NEAREST`  | 最近点 |
| 1 | `BILINEAR` | 双线性的 |
| 2 | `BICUBIC`  | 双三次的 |


---
### Wrap
```cpp
enum Wrap {
    CLAMP_TO_EDGE  = 0,
    ZERO           = 1,
    REPEAT         = 2
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `CLAMP_TO_EDGE` | 固定边缘的 |
| 1 | `ZERO`          | 零 |
| 1 | `REPEAT`        | 重复 |

## 成员函数

---
### create
```cpp
static ImageProcess* create(const Config& config, const Tensor* dstTensor = nullptr);
```
为给定张量创建给定配置的图像处理

参数：
- `config` 给定配置
- `dstTensor` 给定张量，默认为nullptr

返回：图像处理器

---
### create
```cpp
static ImageProcess* create(const ImageFormat sourceFormat = RGBA, const ImageFormat destFormat = RGBA,
                            const float* means = nullptr, const int meanCount = 0, const float* normals = nullptr,
                            const int normalCount = 0, const Tensor* dstTensor = nullptr);
```
为给定张量创建给定配置的图像处理

参数：
- `sourceFormat` 源图像格式，默认为RGBA
- `destFormat` 目的图像格式，默认为RGBA
- `means` 给定方法，默认为nullptr
- `meanCount` 给定方法数量，默认为0
- `normals` 给定常量，默认为nullptr
- `normalCount` 给定常量数量，默认为0
- `dstTensor` 给定张量，默认为nullptr

返回：图像处理器

---
### ~ImageProcess
析构函数

---
### destroy
```cpp
static void destroy(ImageProcess* imageProcess);
```
释放图像资源

参数：
- `imageProcess` 被释放的图像进程

返回：`void`

---
### matrix
```cpp
inline const Matrix& matrix() const {
        return mTransform;
    };
```
得到仿射变换矩阵

参数：无

返回：仿射变换矩阵

---
### setMatrix
```cpp
void setMatrix(const Matrix& matrix);
```
设置仿射变换矩阵

参数：
- `matrix` 源仿射变换矩阵

返回：`void`

---
### convert
```cpp
ErrorCode convert(const uint8_t* source, int iw, int ih, int stride, Tensor* dest);
```
将源数据转换为给定的张量

参数：
- `source` 源资源数据
- `iw` 源资源数据的宽度
- `ih` 源资源数据的高度
- `stride` 每行的元素数，100宽RGB包含至少300个元素
- `dest` 目的张量

返回：结果code

---
### convert
```cpp
ErrorCode convert(const uint8_t* source, int iw, int ih, int stride, void* dest, int ow, int oh, int outputBpp = 0,
                  int outputStride = 0, halide_type_t type = halide_type_of<float>());
```
将源数据转换为给定的张量

参数：
- `source` 源资源数据
- `iw` 源资源数据的宽度
- `ih` 源资源数据的高度
- `stride` 每行的元素数，100宽RGB包含至少300个元素
- `dest` 目的张量
- `ow` 输出宽度
- `oh` 输出高度
- `outputBpp` 如果是0，设置为保存和config.destFormat，默认为0
- `outputStride` 如果为0，设置为ow * outputBpp，默认为0
- `type` 支持`halide_type_of<uint8_t>`和`halide_type_of<float>`，默认为`halide_type_of<float>`

返回：结果code

---
### createImageTensor
```cpp
template <typename T>
static Tensor* createImageTensor(int w, int h, int bpp, void* p = nullptr) {
    return createImageTensor(halide_type_of<T>(), w, h, bpp, p);
}
static Tensor* createImageTensor(halide_type_t type, int w, int h, int bpp, void* p = nullptr);
```
用给定的数据创建张量

参数：
- `type` 只支持halide_type_of和halide_type_of
- `w` 图像宽度
- `h` 图像高度
- `bpp` 每像素字节
- `p` 像素数据指针，默认为nullptr

返回：创建的张量

---
### setPadding
```cpp
void setPadding(uint8_t value) {
    mPaddingValue = value;
};
```
当wrap=ZERO时设置填充值

参数：
- `value` 填充值

返回：`void`

---
### setDraw
```cpp
void setDraw();
```
设置绘制模式

参数：无

返回：`void`

---
### draw
```cpp
void draw(uint8_t* img, int w, int h, int c, const int* regions, int num, const uint8_t* color);
```
绘制img区域的颜色

参数：
- `img` 要绘制的图像
- `w` 图像的宽度
- `h` 图像的高度
- `c` 图像的通道
- `regions` 要绘制的区域，大小为[num * 3]包含num x {y, xl, xr}
- `num` 区域数量
- `color` 要绘制的颜色

返回：`void`