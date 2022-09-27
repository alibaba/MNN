# Tensor
```cpp
class Tensor
```

## 枚举类
### DimensionType
```cpp
用于创建张量的维度类型
enum DimensionType {
    TENSORFLOW,
    CAFFE,
    CAFFE_C4
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `TENSORFLOW` | tensorflow网络类型，数据格式为NHWC |
| 1 | `CAFFE` | caffe网络类型，数据格式为NCHW |
| 2 | `CAFFE_C4` | caffe网络类型，数据格式为NC4HW4 |


---
数据处理类型
### HandleDataType
```cpp
enum HandleDataType {
    HANDLE_NONE        = 0,
    HANDLE_STRING      = 1
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `HANDLE_NONE` | 默认处理类型  |
| 1 | `HANDLE_STRING` | 字符串处理类型 |


---
### MapType
```cpp
张量映射类型：读或写
enum MapType {
    MAP_TENSOR_WRITE   = 0,
    MAP_TENSOR_READ    = 1
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `MAP_TENSOR_WRITE` | 默认写类型 |
| 1 | `MAP_TENSOR_READ` | 读类型 |

## 成员函数

---
### Tensor
构造函数
```cpp
Tensor(int dimSize = 4, DimensionType type = CAFFE);
```
创建一个具有维度大小和类型的张量，而不需要为数据获取内存

参数：
- `dimSize` 尺寸大小，默认为4
- `type` 张量的维度类型，默认为CAFFE

返回：具有维度大小和类型的张量

---
### Tensor
构造函数
```cpp
Tensor(const Tensor* tensor, DimensionType type = CAFFE, bool allocMemory = true);
```
创建一个与给定张量形状相同的张量

参数：
- `tensor` 形状提供者
- `type` 张量的维度类型，默认为CAFFE
- `allocMemory` 是否为数据获取内存

返回：给定张量形状相同的张量

---
### ~Tensor
析构函数

---
### createDevice
```cpp
static Tensor* createDevice(const std::vector<int>& shape, halide_type_t type, DimensionType dimType = TENSORFLOW);
```
创建具有形状、数据类型和维度类型的张量，存储数据的内存不会被获取，调用后端的onAcquireBuffer来准备内存

参数：
- `shape` 张量的形状
- `type` 数据类型
- `dimType` 张量的维度类型，默认为TENSORFLOW

返回：具有形状、数据类型和维度类型的张量

---
### createDevice
```cpp
static Tensor* createDevice(const std::vector<int>& shape, DimensionType dimType = TENSORFLOW) {
    return createDevice(shape, halide_type_of<T>(), dimType);
};
```
创建具有形状和尺寸类型的张量，数据类型用“T”表示，存储数据的内存不会被获取，调用后端的onAcquireBuffer来准备内存

参数：
- `shape` 张量的形状
- `dimType` 张量的维度类型，默认为TENSORFLOW

返回：具有形状、数据类型和维度类型的张量

---
### create
```cpp
static Tensor* create(const std::vector<int>& shape, halide_type_t type, void* data = NULL,
                      DimensionType dimType = TENSORFLOW);
```
创建具有形状、数据类型、数据和维度类型的张量

参数：
- `shape` 张量的形状
- `type` 数据类型
- `data` 数据保存
- `dimType` 张量的维度类型，默认为TENSORFLOW

返回：具有形状、数据类型、数据和维度类型的张量

---
### create
```cpp
static Tensor* create(const std::vector<int>& shape, void* data = NULL, DimensionType dimType = TENSORFLOW) {
    return create(shape, halide_type_of<T>(), data, dimType);
};
```
创建具有形状、数据和维度类型的张量，数据类型用‘T’表示

参数：
- `shape` 张量的形状
- `data` 数据保存
- `dimType` 张量的维度类型，默认为TENSORFLOW

返回：具有形状、数据类型、数据和维度类型的张量

---
### clone
```cpp
static Tensor* clone(const Tensor* src, bool deepCopy = false);
```
拷贝张量

参数：
- `src` 张量
- `deepCopy` 是否创建新的内容和复制，目前只支持deepCopy = false

返回：拷贝的张量

---
### destroy
```cpp
static void destroy(Tensor* tensor);
```
释放张量

参数：
- `tensor` 需要释放的张量对象

返回：`void`

---
### copyFromHostTensor
```cpp
bool copyFromHostTensor(const Tensor* hostTensor);
```
对于DEVICE张量，从给定的HOST张量拷贝数据

参数：
- `hostTensor` HOST张量，数据提供者

返回：DEVICE张量为真，HOST张量为假

---
### copyToHostTensor
```cpp
bool copyToHostTensor(Tensor* hostTensor) const;
```
对于DEVICE张量，将数据复制到给定的HOST张量

参数：
- `hostTensor` HOST张量，数据消费者

返回：DEVICE张量为真，HOST张量为假

---
### createHostTensorFromDevice
```cpp
static Tensor* createHostTensorFromDevice(const Tensor* deviceTensor, bool copyData = true);
```
从DEVICE张量创建HOST张量，可以复制数据也可以不复制数据

参数：
- `deviceTensor` DEVICE张量
- `copyData` 是否复制数据，默认为true

返回：HOST张量

---
### getDimensionType
```cpp
DimensionType getDimensionType() const;
```
获取维度类型

参数：无

返回：维度类型

---
### getHandleDataType
```cpp
HandleDataType getHandleDataType() const;
```
处理数据类型，当数据类型代码为halide_type_handle时使用

参数：无

返回：处理数据类型

---
### setType
```cpp
void setType(int type);
```
设置数据类型

参数：
- `type` 定义在“Type_generated.h”中的数据类型

返回：`void`

---
### getType
```cpp
inline halide_type_t getType() const {
    return mBuffer.type;
};
```
获取数据类型

参数：无

返回：数据类型

---
### host
```cpp
template <typename T>
T* host() const {
    return (T*)mBuffer.host;
};
```
访问Host内存，数据类型用“T”表示

参数：无

返回：“T”类型的数据点

---
### deviceId
```cpp
uint64_t deviceId() const {
    return mBuffer.device;
};
```
访问设备内存

参数：无

返回：设备数据ID，ID的含义因后端而异

---
### dimensions
```cpp
int dimensions() const {
    return mBuffer.dimensions;
};
```
张量维度

参数：无

返回：维度

---
### shape
```cpp
std::vector<int> shape() const;
```
得到所有维度的范围

参数：无

返回：维度的程度

---
### size
```cpp
int size() const;
```
考虑到重新排序标志，计算存储数据所需的字节数

参数：无

返回：存储数据所需的字节数

---
### elementSize
```cpp
inline int elementSize() const {
    return size() / mBuffer.type.bytes();
};
```
考虑到重新排序标志，计算存储数据所需的元素数量

参数：无

返回：存储数据所需的元素数量

---
### width
```cpp
inline int width() const {
    if (getDimensionType() == TENSORFLOW) {
        return mBuffer.dim[2].extent;
    }
    return mBuffer.dim[3].extent;
};
```
张量宽度

参数：无

返回：张量宽度

---
### height
```cpp
inline int height() const {
    if (getDimensionType() == TENSORFLOW) {
        return mBuffer.dim[1].extent;
    }
    return mBuffer.dim[2].extent;
};
```
张量高度

参数：无

返回：张量高度

---
### channel
```cpp
inline int channel() const {
    if (getDimensionType() == TENSORFLOW) {
        return mBuffer.dim[3].extent;
    }
    return mBuffer.dim[1].extent;
};
```
张量通道

参数：无

返回：张量通道

---
### batch
```cpp
inline int batch() const {
    return mBuffer.dim[0].extent;
};
```
张量批量

参数：无

返回：张量批量

---
### stride
```cpp
inline int stride(int index) const {
    return mBuffer.dim[index].stride;
};
```
返回张量的步幅

参数：
- `index` 指定维度

返回：张量的步幅

---
### length
```cpp
inline int length(int index) const {
    return mBuffer.dim[index].extent;
};
```
返回张量的长度

参数：
- `index` 指定维度

返回：张量的长度

---
### setStride
```cpp
inline void setStride(int index, int stride) {
    mBuffer.dim[index].stride = stride;
};
```
设置张量的步幅

参数：
- `index` 指定维度
- `stride` 步幅

返回：`void`

---
### setLength
```cpp
inline void setLength(int index, int length) {
    mBuffer.dim[index].extent = length;
};
```
设置张量的长度

参数：
- `index` 指定维度
- `stride` 长度

返回：`void`

---
### print
```cpp
void print() const;
```
打印张量数据，仅供调试使用

参数：无

返回：`void`

---
### printShape
```cpp
void printShape() const;
```
打印张量的形状

参数：无

返回：`void`

---
### map
```cpp
void* map(MapType mtype, DimensionType dtype);
```
GPU张量，以获得主机ptr

参数：
- `mtype` 张量映射类型：读或写
- `dtype` 张量类型

返回：主机ptr

---
### unmap
```cpp
void unmap(MapType mtype, DimensionType dtype, void* mapPtr);
```
GPU张量

参数：
- `mtype` 张量映射类型：读或写
- `dtype` 张量类型
- `mapPtr` 主机ptr

返回：`void`

---
### wait
```cpp
int wait(MapType mtype, bool finish);
```
等待直到张量准备好读/写

参数：
- `mtype` 等待读取或写入
- `finish` 等待命令刷新或完成

返回：读/写