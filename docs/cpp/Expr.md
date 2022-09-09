# VARP
```cpp
class VARP
```

## 枚举类
### Dimensionformat
```cpp
enum Dimensionformat {
    NHWC,
    NC4HW4,
    NCHW
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `NHWC` |  |
| 1 | `NC4HW4` | |
| 2 | `NCHW` |  |

---
### InputType
```cpp
enum InputType {
    INPUT = 0,
    CONSTANT = 1,
    TRAINABLE = 2,
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `INPUT` | 默认输入变量 |
| 1 | `CONSTANT` | 常量 |
| 2 | `TRAINABLE` | 可训练变量 |

## 成员函数

---
### VARP
```cpp
VARP(std::shared_ptr<Variable> c) {
    mContent = std::move(c);
};
```
调用移动构造函数，将c的对象移动至mContent对象


参数：
- `c` Variable类型变量

返回：`void`

---
### VARP
```cpp
VARP(Variable* c) {
    mContent.reset(c);
};
```
重置mContent对象中的Variable对象


参数：
- `c` Variable类型变量

返回：`void`

---
### get
```cpp
Variable* get() const  {
    return mContent.get();
};
```
获取mContent的Variable对象


参数：无

返回：Variable对象

---
### VARP
```cpp
VARP(const VARP& var) {
    mContent = var.mContent;
};
```
把var的content对象赋值给mContent变量


参数：
- `var` 输入变量

返回：`void`

---
### VARP
```cpp
VARP(VARP&& var) {
    mContent = std::move(var.mContent);
};
```
调用移动构造函数，将var.mContent的对象移动至mContent对象


参数：
- `var` 输入变量

返回：`void`

---
### operator+
```cpp
VARP operator+(VARP var) const;
```
VARP类型对象加法计算


参数：
- `var` 输入变量

返回：VARP对象

---
### operator-
```cpp
VARP operator-(VARP var) const;
```
VARP类型对象减法计算


参数：
- `var` 输入变量

返回：VARP对象

---
### operator*
```cpp
VARP operator*(VARP var) const;
```
VARP类型对象乘法计算


参数：
- `var` 输入变量

返回：VARP对象

---
### operator/
```cpp
VARP operator/(VARP var) const;
```
VARP类型对象除法计算


参数：
- `var` 输入变量

返回：VARP对象

---
### mean
```cpp
VARP mean(INTS dims) const;
```
计算均值


参数：
- `dims` 一个向量

返回：VARP对象

---
### sum
```cpp
VARP sum(INTS dims) const;
```
计算和


参数：
- `dims` 一个向量

返回：VARP对象

---
### operator==
```cpp
bool operator==(const VARP& var) const {
    return var.mContent == mContent;
};
```
重载相等运算符，判断var.mContent和mContent是否相等


参数：
- `var` 输入变量

返回：true/false

---
### operator<
```cpp
bool operator<(const VARP& var) const {
    return mContent < var.mContent;
};
```
重载小于运算符，判断mContent是否小于var.mContent


参数：
- `var` 输入变量

返回：true/false

---
### operator<=
```cpp
bool operator<=(const VARP& var) const {
    return mContent <= var.mContent;
};
```
重载小于等于运算符，判断mContent是否小于等于var.mContent


参数：
- `var` 输入变量

返回：true/false

---
### operator=
```cpp
VARP& operator=(const VARP& var) {
    mContent = var.mContent;
    return *this;
};
```
拷贝var.mContent对象到mContent


参数：
- `var` 输入变量

返回：当前对象的拷贝

---
### operator=
```cpp
VARP& operator=(Variable* var) {
    mContent.reset(var);
    return *this;
};
```
重置mContent对象中的Variable对象，并拷贝



参数：
- `var` 输入变量

返回：当前对象的拷贝

---
### operator->
```cpp
Variable* operator->() const  {
    return mContent.get();
};
```
获取mContent对象的值


参数：无

返回：Variable对象

---
### fix
```cpp
bool fix(InputType type) const;
```
朝零方向取整


参数：
- `type` 输入数据类型

返回：true/false

---
### operator==
```cpp
inline bool operator==(Variable* src, VARP dst) {
    return src == dst.get();
};
```
重载相等运算符，判断src和dst.get()是否相等


参数：
- `src` Variable类型输入变量
- `dst` VARP类型输入变量

返回：true/false

---
### operator!=
```cpp
inline bool operator!=(Variable* src, VARP dst) {
    return src != dst.get();
};
```
重载不相等运算符，判断src和dst.get()是否不相等


参数：
- `src` Variable类型输入变量
- `dst` VARP类型输入变量

返回：true/false


# Variable
```cpp
class Variable
```
## 成员函数
---
### name
```cpp
const std::string& name() const;
```
获取Variable对象的名称


参数：无

返回：名称

---
### setName
```cpp
void setName(const std::string& name);
```
设置名称


参数：
- `name` 名称

返回：`void` 

---
### expr
```cpp
std::pair<EXPRP, int> expr() const {
    return std::make_pair(mFrom, mFromIndex);
};
```
创建一个EXPRP对象，需要mFrom作为参数，位置为mFromIndex


参数：无

返回：EXPRP对象

---
### getInfo
```cpp
const Info* getInfo();
```
获取Variable对象的相关信息


参数：无

返回：如果计算信息错误，返回nullptr

---
### resize
```cpp
bool resize(INTS dims);
```
调整Variable对象的大小


参数：
- `dims` 一个向量

返回：true/false

---
### readMap
```cpp
template <typename T>
const T* readMap() {
    return (const T*)readInternal();
};
```
读取内部信息



参数：无

返回：信息

---
### writeMap
```cpp
template <typename T>
T* writeMap() {
    return (T*)writeInternal();
};
```
写入内部信息


参数：无

返回：信息

---
### input
```cpp
bool input(VARP src);
```
输入信息


参数：
- `src` 输入变量

返回：true/false

---
### replace
```cpp
static void replace(VARP dst, VARP src);
```
替换信息


参数：
- `dst` 输入变量
- `src` 输入变量

返回：`void`

---
### create
```cpp
static VARP create(EXPRP expr, int index = 0);
```
在index位置创建expr对象


参数：
- `expr` 输入变量
- `index` 位置下标，默认为0

返回：VARP对象

---
### load
```cpp
static std::vector<VARP> load(const char* fileName);
```
通过文件名加载对象


参数：
- `fileName` 文件名

返回：VARP对象矩阵

---
### loadMap
```cpp
static std::map<std::string, VARP> loadMap(const char* fileName);
```
通过文件名读取模型对象


参数：
- `fileName` 文件名

返回：模型对象

---
### load
```cpp
static std::vector<VARP> load(const uint8_t* buffer, size_t length);
```
加载存储的，长度为length的模型对象


参数：
- `buffer` 存储数据
- `length` 数据长度

返回：模型对象

---
### loadMap
```cpp
static std::map<std::string, VARP> loadMap(const uint8_t* buffer, size_t length);
```
读取存储的，长度为length的模型对象


参数：
- `buffer` 存储数据
- `length` 数据长度 

返回：模型对象

---
### getInputAndOutput
```cpp
static std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> getInputAndOutput(const std::map<std::string, VARP>& allVariable);
```
获取模型输入输出节点


参数：
- `allVariable` 输入变量

返回：模型输入输出节点

---
### mapToSequence
```cpp
static std::vector<VARP> mapToSequence(const std::map<std::string, VARP>& source);
```
模型的输出节点及其名称


参数：
- `source` 输入变量

返回：输出节点及其名称

---
### getExecuteOrder
```cpp
static std::vector<EXPRP> getExecuteOrder(const std::vector<VARP>& output);
```
获取操作指令


参数：
- `output` 输入变量

返回：指令集

---
### save
```cpp
static void save(const std::vector<VARP>& vars, const char* fileName);
```
保存vars到指定位置


参数：
- `vars` 输入变量
- `fileName` 文件名

返回：`void`

---
### save
```cpp
static std::vector<int8_t> save(const std::vector<VARP>& vars);
```
保存vars到默认位置


参数：
- `vars` 输入变量

返回：`void`

---
### save
```cpp
static void save(const std::vector<VARP>& vars, NetT* dest);
```
保存vars到dest位置


参数：
- `vars` 输入变量
- `dest` 目标地址

返回：`void`

---
### prepareCompute
```cpp
static void prepareCompute(const std::vector<VARP>& vars, bool forceCPU = false);
```
将几个变量打包在一个管道中进行计算


参数：
- `vars` 输入变量
- `forceCPU` 是否强制使用CPU，默认为false

返回：`void`

---
### compute
```cpp
static void compute(const std::vector<VARP>& vars, bool forceCPU = false);
```
计算变量


参数：
- `vars` 输入变量
- `forceCPU` 是否强制使用CPU，默认为false

返回：`void`

---
### linkNumber
```cpp
size_t linkNumber() const;
```
获取输出信息的size


参数：无

返回：size

---
### toExprs
```cpp
const std::vector<WeakEXPRP>& toExprs() const;
```
返回模型对象信息


参数：无

返回：模型对象信息

---
### setExpr
```cpp
void setExpr(EXPRP expr, int index) {
    mFrom = expr;
    mFromIndex = index;
};
```
在index位置设置EXPRP对象


参数：无

返回：EXPRP对象


# Expr
```cpp
class Expr
```

## 枚举类
---
### MemoryType
```cpp
enum MemoryType {
    COPY,
    MOVE,
    REF
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `COPY` | 拷贝 |
| 1 | `MOVE` | 移动 |
| 2 | `REF` | 引用 |

## 成员函数
---
### create
```cpp
static EXPRP create(Tensor* tensor, bool own = false);
```
创建一个包含tensor的EXPRP变量


参数：
- `tensor` 输入变量
- `own` 默认为false

返回：EXPRP变量

---
### create
```cpp
static EXPRP create(Variable::Info&& info, const void* ptr, VARP::InputType type, MemoryType copy = COPY);
```
创建一个包含info对象的EXPRP变量


参数：
- `info` Variable类型输入变量
- `ptr` 目标对象地址
- `type` 输入数据类型
- `copy` 内存拷贝

返回：EXPRP变量

---
### create
```cpp
static EXPRP create(const OpT* op, std::vector<VARP> inputs, int outputSize = 1);
```
创建EXPRP变量


参数：
- `op` 输入变量
- `inputs` 输入变量
- `outputSize` 输出信息大小，默认为1

返回：EXPRP变量

---
### create
```cpp
static EXPRP create(std::shared_ptr<BufferStorage> extra, std::vector<VARP>&& inputs, int outputSize = 1);
```
创建EXPRP变量


参数：
- `extra` 输入变量
- `inputs` 输入变量
- `outputSize` 输出信息大小，默认为1

返回：EXPRP变量

---
### create
```cpp
static EXPRP create(std::unique_ptr<OpT>&& op, std::vector<VARP> inputs, int outputSize = 1) {
    return create(op.get(), inputs, outputSize);
};
```
创建EXPRP变量


参数：
- `op` 输入变量
- `inputs` 输入变量
- `outputSize` 输出信息大小，默认为1

返回：EXPRP变量

---
### setName
```cpp
void setName(const std::string& name);
```
设置名称


参数：
- `name` 名称

返回：`void`

---
### get
```cpp
const Op* get() const {
    return mOp;
};
```
获取对象信息


参数：无

返回：对象信息

---
### inputs
```cpp
const std::vector<VARP>& inputs() const {
    return mInputs;
};
```
获取输入节点信息


参数：无

返回：输入节点信息

---
### outputSize
```cpp
int outputSize() const {
    return (int)mOutputNames.size();
};
```
返回输出节点信息大小


参数：无

返回：size

---
### replace
```cpp
static void replace(EXPRP oldExpr, EXPRP newExpr);
```
用newExpr替换oldExpr


参数：
- `oldExpr` 输入变量，源对象
- `newExpr` 输入变量，目标对象

返回：`void`

---
### requireInfo
```cpp
bool requireInfo();
```
获取需要的信息


参数：无 

返回：信息

---
### visitOutputs
```cpp
void visitOutputs(const std::function<bool(EXPRP, int)>& visit);
```
访问输出节点的信息


参数：
- `visit` 访问方法

返回：`void`

---
### visit
```cpp
static void visit(EXPRP expr, const std::function<bool(EXPRP)>& before, const std::function<bool(EXPRP)>& after);
```
访问某一个范围的信息


参数：
- `expr` 输入变量
- `before` before指针
- `after`  after指针

返回：`void`

---
### outputs
```cpp
const std::vector<WeakEXPRP>& outputs() const {
    return mTo;
};
```
返回输出节点信息


参数：无

返回：信息

---
### ~Expr()
析构函数

---
### visited
```cpp
bool visited() const {
    return mVisited;
};
```
是否已经访问过


参数：无

返回：true/false

---
### setVisited
```cpp
void setVisited(bool visited) {
    mVisited = visited;
};
```
设置已经被访问


参数：
- `visited` 是否访问

返回：`void`

---
### name
```cpp
const std::string& name() const {
    return mName;
};
```
获取名称


参数：无

返回：名称

---
### outputName
```cpp
const std::string& outputName(int index) {
    return mOutputNames[index];
};
```
输出指定index的名称


参数：
- `index` 下标

返回：名称

---
### inputType
```cpp
VARP::InputType inputType() const {return mType;};
```
返回当前输入类型


参数：无

返回：输入类型

---
### outputInfo
```cpp
Variable::Info* outputInfo(int index) const;
```
返回指定下标的输出信息


参数：
- `index` 下标值

返回：输出信息

---
### extra
```cpp
std::shared_ptr<BufferStorage> extra() const {
    return mStorage;
};
```
返回附加信息


参数：无

返回：附加信息

---
### inside
```cpp
std::shared_ptr<Inside> inside() const {
    return mInside;
};
```
返回内部信息


参数：无

返回：内部信息

---
### valid
```cpp
bool valid() const {
    return mValid;
};
```
是否有效


参数：无

返回：true/false
