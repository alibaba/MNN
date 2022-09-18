# Module
```cpp
class Module
```

## 成员函数
---
### Tensor
构造函数
```cpp
Module() == default;
```
创建一个空Module

参数：无

返回：Module对象

---
### ~Module
析构函数
```cpp
virtual ~Module() == default;
```

---
### onForward
```cpp
virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) = 0;
```
模块前向传播,返回多个结果变量

参数：
- `inputs` 前向传播输入变量

返回：前向传播输出变量

---
### forward
```cpp
Express::VARP forward(Express::VARP input);
```
模块前向传播,返回一个结果变量

参数：
- `input` 前向传播输入变量

返回：前向传播输出变量

---
### parameters
```cpp
std::vector<Express::VARP> parameters() const;
```
获取Module的参数

参数：无

返回：Module的参数

---
### loadParameters
```cpp
bool loadParameters(const std::vector<Express::VARP>& parameters);
```
加载现有的参数

参数：
- `parameters` 参数值

返回：是否成功加载参数

---
### setIsTraining
```cpp
void setIsTraining(const bool isTraining);
```
设置Module的训练状态

参数：
- `isTraining` 是否为训练模式

返回：`void`

---
### getIsTraining
```cpp
bool getIsTraining();
```
获取_Module的是否为训练模式

参数：无

返回：Module是否为训练模式，是则返回true，不是返回false

---
### clearCache
```cpp
void clearCache();
```
清除Module的缓存，并递归清除子模块的缓存

参数：无

返回：`void`

---
### name
```cpp
const std::string& name() const {
    return mName;
};
```
获取Module的名称

参数：无

返回：Module的名称

---
### setName
```cpp
void setName(std::string name) {
    mName = std::move(name);
};
```
设置Module的名称

参数：
- `name` 模块的名称

返回：`void`

---
### type
```cpp
const std::string type() const {
    return mType;
};
```
获取Module的类型

参数：无

返回：Module的类型

---
### setType
```cpp
void setType(std::string type) {
    mType = std::move(type);
};
```
设置Module的类型

参数：
- `type` 模块的类型

返回：`void`

---
### addParameter
```cpp
int addParameter(Express::VARP parameter);
```
添加参数

参数：
- `parameter` 参数值

返回：添加前的参数数量

---
### setParameter
```cpp
void setParameter(Express::VARP parameter, int index);
```
设置参数

参数：
- `type` 参数值
- `index` 参数的位置索引

返回：`void`

---
### createEmpty
```cpp
static Module* createEmpty(const std::vector<Express::VARP>& parameters);
```
根据参数创建一个空的Module对象

参数：
- `parameters` 参数值

返回：创建的空的Module对象

---
### load
```cpp
static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const Config* config = nullptr);
```
加载module对象

参数：
- `inputs` module输入信息
- `outputs` module输出信息
- `buffer` 缓冲信息
- `length` 信息长度
- `config` 其他配置项

返回：module对象

---
### load
```cpp
static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const char* fileName, const Config* config = nullptr);
```
加载module对象

参数：
- `inputs` module输入信息
- `outputs` module输出信息
- `fileName` 文件名
- `config` 其他配置项

返回：module对象

---
### load
```cpp
static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const char* fileName, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Config* config = nullptr);
```
加载module对象

参数：
- `inputs` module输入信息
- `outputs` module输出信息
- `fileName` 文件名
- `rtMgr` 运行时资源
- `config` 其他配置项

返回：module对象

---
### load
```cpp
static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Config* config = nullptr);
```
加载module对象

参数：
- `inputs` module输入信息
- `outputs` module输出信息
- `buffer` 缓冲信息
- `length` 信息长度
- `rtMgr` 运行时资源
- `config` 其他配置项

返回：module对象

---
### load
```cpp
static Module* extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain, const std::map<std::string, SubGraph>& subGraph = {});
```
加载module对象

参数：
- `inputs` module输入信息
- `outputs` module输出信息
- `fortrain` 
- `subGraph` 子图

返回：module对象

---
### clone
```cpp
static Module* clone(const Module* module, const bool shareParams = false);
```
克隆Module对象

参数：
- `module` module对象实例
- `shareParams` 是否共享参数，默认为false

返回：Module对象实例

---
### getInfo
```cpp
const Info* getInfo() const;
```
获取Module的信息

参数：无

返回：Module的信息

---
### CloneContext
```cpp
CloneContext() = default;
```
克隆Module的内容

参数：无

返回：Module的内容

---
### CloneContext
```cpp
explicit CloneContext(const bool shareParams)
        : mShareParams(shareParams) {};
```
克隆Module的内容

参数：
- `shareParams` 是否共享参数

返回：Module的内容

---
### ~CloneContext
析构函数
```cpp
virtual ~CloneContext() = default;
```

---
### shareParams
```cpp
const bool shareParams() const { return mShareParams; };
```
是否共享参数

参数：无

返回：共享返回true，反之则为false

---
### getOrClone
```cpp
EXPRP getOrClone(const EXPRP expr);
```
获取克隆的EXPRP对象

参数：
- `expr` EXPRP对象值

返回：EXPRP对象

---
### getOrClone
```cpp
VARP getOrClone(const VARP var);
```
获取克隆的VARP对象

参数：
- `expr` VARP对象值

返回：VARP对象

---
### clone
```cpp
virtual Module* clone(CloneContext* ctx) const {
    return nullptr;
};
```
克隆Module对象

参数：
- `ctx` 克隆的上下文

返回：Module对象

---
### registerModel
```cpp
void registerModel(const std::vector<std::shared_ptr<Module>>& children);
```
注册子模块

参数：
- `children` 子模块列表

返回：`void`

---
### destroy
```cpp
static void destroy(Module* m);
```
销毁Module对象

参数：
- `m` Module对象

返回：`void`