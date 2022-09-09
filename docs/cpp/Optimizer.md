# Optimizer
```cpp
class Optimizer
```
## 枚举类
### Device
```cpp
enum Device {
    CPU = 0,
    GPU = 1,
    OTHER = 2,
    AUTO = 3
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `CPU` | 中央处理器 |
| 1 | `GPU` | 图像处理器 |
| 2 | `OTHER` | 其他 |
| 3 | `AUTO` | 自定义 |

## 成员函数
---
### Optimizer
构造函数
```cpp
Optimizer() = default;
```
创建一个空Optimizer

参数：无

返回：Optimizer对象

---
### ~Optimizer
析构函数
```cpp
virtual ~Optimizer() = default;
```
创建一个空Optimizer

参数：无

返回：Optimizer对象

---
### create
```cpp
static std::shared_ptr<Optimizer> create(Config config);
```
创建一个Optimizer对象

参数：
- `config` 配置信息，包括线程、Device和MNNForwardType等信息

返回：Optimizer对象

---
### onGetParameters
```cpp
virtual std::shared_ptr<Parameters> onGetParameters(const std::vector<VARP>& outputs) {
    return nullptr;
};
```
获取Optimizer对象的参数

参数：
- `outputs` Optimizer输出信息

返回：Optimizer对象的参数

---
### onMeasure
```cpp
virtual Cost onMeasure(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr) = 0;
```
返回Cost对象信息，包括compute(计算)和memory(内存)信息，parameters必须与onGetParameters相同

参数：
- `outputs` Optimizer输出信息
- `parameters` 与onGetParameters相同

返回：Cost对象信息

---
### onExecute
```cpp
virtual bool onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr) = 0;
```
修改输出信息，parameters必须与onGetParameters相同

参数：
- `outputs` Optimizer输出信息
- `parameters` 与onGetParameters相同

返回：是否修改输出成功

## Parameters
```cpp
class Parameters
```
## 成员函数

---
### Parameters
```cpp
Parameters(int n);
```
创建一个Parameters对象

参数：
- `n` 成员个数

返回：Parameters对象

---
### ~Parameters
析构函数

---
### get
```cpp
float* get() const {
    return mValue;
};
```
获取Parameters对象成员数量

参数：无

返回：Parameters对象成员数量

---
### size
```cpp
int size() const {
    return mSize;
};
```
获取Parameters对象大小

参数：无

返回：Parameters对象大小


