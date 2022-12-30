# Interpreter
```cpp
class Interpreter
```

## 枚举类
### SessionMode
```cpp
enum SessionMode {
    Session_Debug = 0,
    Session_Release = 1,
    Session_Input_Inside = 2,
    Session_Input_User = 3,
    Session_Output_Inside = 4,
    Session_Output_User = 5,
    Session_Resize_Direct = 6,
    Session_Resize_Defer = 7,
    Session_Backend_Fix = 8,
    Session_Backend_Auto = 9,
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `Session_Debug` | 可以执行callback函数，并获取Op信息(*默认*) |
| 1 | `Session_Release` | 不可执行callback函数 |
| 2 | `Session_Input_Inside` | 输入由session申请(*默认*) |
| 3 | `Session_Input_User` | 输入由用户申请 |
| 4 | `Session_Output_Inside` | 输出依赖于session不可单独使用 |
| 5 | `Session_Output_User` | 输出不依赖于session可单独使用 |
| 6 | `Session_Resize_Direct` | 在创建Session时执行resize(*默认*) |
| 7 | `Session_Resize_Defer` | 在创建Session时不执行resize |
| 8 | `Session_Backend_Fix` | 使用用户指定的后端，后端不支持时回退CPU |
| 9 | `Session_Backend_Auto` | 根据算子类型自动选择后端 |

---
### ErrorCode
```cpp
enum ErrorCode {
    NO_ERROR           = 0,
    OUT_OF_MEMORY      = 1,
    NOT_SUPPORT        = 2,
    COMPUTE_SIZE_ERROR = 3,
    NO_EXECUTION       = 4,
    INVALID_VALUE      = 5,
    INPUT_DATA_ERROR = 10,
    CALL_BACK_STOP   = 11,
    TENSOR_NOT_SUPPORT = 20,
    TENSOR_NEED_DIVIDE = 21,
};
```

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `NO_ERROR` | 没有错误，执行成功  |
| 1 | `OUT_OF_MEMORY` | 内存不足，无法申请内存 |
| 2 | `NOT_SUPPORT` | 有不支持的OP |
| 3 | `COMPUTE_SIZE_ERROR` | 形状计算出错 |
| 4 | `NO_EXECUTION` | 创建执行时出错 |
| 10 | `INPUT_DATA_ERROR` | 输入数据出错 |
| 11 | `CALL_BACK_STOP` | 用户callback函数退出 |
| 20 | `TENSOR_NOT_SUPPORT` | resize出错 |
| 21 | `TENSOR_NEED_DIVIDE` | resize出错 |

---
### SessionInfoCode
```cpp
enum SessionInfoCode {
    MEMORY = 0,
    FLOPS = 1,
    BACKENDS = 2,
    RESIZE_STATUS = 3,
    ALL
};
```

| value | name |      说明         |
|:-----|:-----|:-----------------|
| 0 | `MEMORY` | 会话的内存占用大小，MB计算，浮点类型数据 |
| 1 | `FLOPS` | 会话的计算量，flops，浮点数据类型 |
| 2 | `BACKENDS` | 会话的后端数目，个数是config数量加1 |
| 3 | `RESIZE_STATUS` | resize的状态，数据是int类型，0表示就绪，1表示需要分配内存，2表示需要resize |
|   | `ALL` | 以上所有信息 |

---
### HintMode
```cpp
enum HintMode {
    MAX_TUNING_NUMBER = 0,
};
```

| value | name |      说明         |
|:-----|:-----|:-----------------|
| 0 | `MAX_TUNING_NUMBER` | GPU下tuning的最大OP数 |

## 成员函数

---
### Interpreter
该构造函数禁止使用，创建对象请使用`createFromFile`

---
### ~Interpreter
析构函数

---
### createFromFile
```cpp
static Interpreter* createFromFile(const char* file);
```
从文件加载模型，加载`.mnn`模型文件创建一个MNN解释器，返回一个解释器对象

参数：
- `file` MNN模型所放置的完整文件路径

返回：创建成功则返回创建的解释器对象指针，失败就返回`nullptr`

---
### createFromBuffer
```cpp
static Interpreter* createFromBuffer(const void* buffer, size_t size)
```
从内存加载模型，根据指定内存地址与大小，从内存中创建一个计时器对象

参数：
- `buffer` 模型文件内存中的数据指针
- `size`  模型文件字节数大小

返回：创建成功则返回创建的解释器对象指针，失败就返回`nullptr`

---
### destroy
```cpp
static void destroy(Interpreter* net);
```
释放指定的解释器对象

参数：
- `net` 需要释放的解释器对象

返回：`void`

---
### setSessionMode
```cpp
void setSessionMode(SessionMode mode);
```
设置模型推理时回话的执行模式，参考[SessionMode](./Interpreter.html#sessionmode)

*该函数需在`createSession`前调用*

参数：
- `mode` 回话的执行模式

返回：`void`

---
### setCacheFile
```cpp
void setCacheFile(const char* cacheFile, size_t keySize = 128);
```
设置缓存文件，缓存文件在GPU模式下用来存储Kernel与Op信息；执行该函数，在`runSession`前会从缓存文件中加载信息；在`runSession`后会将相关信息写入缓存文件中

*该函数需在`createSession`前调用*

参数：
- `cacheFile` 缓存文件名
- `keySize` 保留参数，现在未使用

返回：`void`

---
### setExternalFile
```cpp
void setExternalFile(const char* file, size_t flag = 128);
```
设置额外文件，额外文件是指存储了模型中权重，常量等数据的文件，在创建`Session`时会从该文件中加载权重等数据。

*该函数需在`createSession`前调用*

参数：
- `file` 额外文件名
- `flag` 保留参数，现在未使用

返回：`void`

---
### updateCacheFile
```cpp
ErrorCode updateCacheFile(Session *session, int flag = 0);
```
更新缓存文件，在最近的一次`resizeSession`中如果修改了缓存信息，就写入到缓存文件中；如果没有修改缓存信息，就什么都不做

*该函数需在`resizeSession`前调用*

参数：
- `session` 要更新的缓存会话
- `flag` 保留参数，现在未使用

返回：更新缓存的[错误码](Interpreter.html#errorcode)

---
### setSessionHint
```cpp
void setSessionHint(HintMode mode, int value);
```
设置会话的额外执行信息，使用[HintMode](Interpreter.html#hintmode)来描述额外信息的类型

*该函数需在`createSession`前调用*

参数：
- `mode` 额外信息的类型
- `value` 额外信息的数值

返回：`void`

---
### createRuntime
```cpp
static RuntimeInfo createRuntime(const std::vector<ScheduleConfig>& configs);
```
根据配置创建一个Runtime，默认情况下，在`createSession`时对应create单独一个Runtime。对于串行的一系列模型，可以先单独创建Runtime，然后在各Session创建时传入，使各模型用共享同样的运行时资源（对CPU而言为线程池、内存池，对GPU而言Kernel池等）；`RuntimeInfo`的定义如下：
```cpp
typedef std::pair<std::map<MNNForwardType, std::shared_ptr<Runtime>>, std::shared_ptr<Runtime>> RuntimeInfo;
```

参数：
- `configs` 调度信息

返回：创建的运行时信息，可以用于创建`Session`

---
### createSession
```cpp
Session* createSession(const ScheduleConfig& config);
Session* createSession(const ScheduleConfig& config, const RuntimeInfo& runtime);
```
根据配置或根据用户指定的运行时信息创建会话`Session`
- 当仅传入`config`时会根据`config`信息创建`Runtime`
- 当传入`runtime`时则使用用户传入的`Runtime`

参数：
- `config` 调度信息
- `runtime` 执行信息

返回：创建的会话`Session`

---
### createMultiPathSession
```cpp
Session* createMultiPathSession(const std::vector<ScheduleConfig>& configs);
Session* createMultiPathSession(const std::vector<ScheduleConfig>& configs, const RuntimeInfo& runtime);
```
根据多个配置创建多段计算路径的会话
- 当仅传入`config`时会根据`config`信息创建`Runtime`
- 当传入`runtime`时则使用用户传入的`Runtime`

参数：
- `configs` 多个调度信息

返回：创建的会话`Session`

---
### releaseSession
```cpp
bool releaseSession(Session* session);
```
释放指定的`Session`

参数：
- `session` 待释放的`Session`

返回：是否成功释放该`Session`

---
### resizeSession
```cpp
void resizeSession(Session* session);
void resizeSession(Session* session, int needRelloc);
```
为session分配内存，进行推理准备工作；该函数一般配合`resizeTensor`一起调用，修改Tensor输入形状后对应整个推理过程中的内存分配也需要修改

参数：
- `session` 改变输入形状后需要重新分配内存的Session对象
- `needRelloc` 是否重新分配内存，如果为0则只进行形状计算不进行内存分配

返回：`void`

---
### releaseModel
```cpp
void releaseModel();
```
当不再需要执行`createSession`和`resizeSession`的时候，可以调用此函数释放解释器中持有的模型资源，会释放模型文件大小的内存

参数：
- `void`

返回：`void`

---
### getModelBuffer
```cpp
std::pair<const void*, size_t> getModelBuffer() const;
```
获取模型模型的内存数据指针和内存大小，方便用户存储模型

参数：
- `void`

返回：内存数据指针和内存大小

---
### getModelVersion
```cpp
const char* getModelVersion() const;
```
获取模型的版本信息，以字符串的形式返回

参数：
- `void`

返回：模型的版本信息，类似：`"2.0.0"`

---
### updateSessionToModel
```cpp
ErrorCode updateSessionToModel(Session* session);
```
将`Session`中`Tensor`的数据更新Model中的常量数据

参数：
- `session` 需要更新的会话

返回：更新数据的[错误码](Interpreter.html#errorcode)

---
### runSession
```cpp
ErrorCode runSession(Session* session) const;
```
运行session执行模型推理，返回对应的error code，需要根据错误码来判断后续是否成功执行模型推理

参数：
- `session` 执行推理的Session对象

返回：执行推理的[错误码](Interpreter.html#errorcode)

---
### runSessionWithCallBack
```cpp
ErrorCode runSessionWithCallBack(const Session* session, const TensorCallBack& before, const TensorCallBack& end, bool sync = false) const;
```
该函数本质上与`runSession`一致，但是提供了用户hook函数的接口，在运行session做网络推理，每层推理前前后会执行的`before`和`end`并根据返回值来决定是否继续执行

参数：
- `session` 执行推理的Session对象
- `before` 每层推理前执行的回调函数，类型为
  ```cpp
  std::function<bool(const std::vector<Tensor*>&, const std::string& /*opName*/)>
  ```
- `end` 每层推理后执行的回调函数，类型同上
- `sync` 是否同步等待执行完成

返回：执行推理的[错误码](Interpreter.html#errorcode)

---
### runSessionWithCallBackInfo
```cpp
ErrorCode runSessionWithCallBackInfo(const Session* session, const TensorCallBackWithInfo& before,
                                     const TensorCallBackWithInfo& end, bool sync = false) const;
```
该函数与`runSessionWithCallBack`相似，但是回调函数中增加了Op的类型和计算量信息，可以用来评估模型的计算量

参数：
- `session` 执行推理的Session对象
- `before` 每层推理前执行的回调函数，类型为
  ```cpp
  std::function<bool(const std::vector<Tensor*>&, const OperatorInfo*)>
  ```
- `end` 每层推理后执行的回调函数，类型同上
- `sync` 是否同步等待执行完成

返回：执行推理的[错误码](Interpreter.html#errorcode)

---
### getSessionInput
```cpp
Tensor* getSessionInput(const Session* session, const char* name);
```
根据`name`返回模型指定会话的输入tensor；如果没有指定tensor名称，则返回第一个输入tensor

参数：
- `session` 持有推理会话数据的Session对象
- `name` Tensor的名称

返回：输入Tensor对象

---
### getSessionOutput
```cpp
Tensor* getSessionOutput(const Session* session, const char* name);
```
根据`name`返回模型指定会话的输出tensor；如果没有指定tensor名称，则返回第一个输出tensor

参数：
- `session` 持有推理会话数据的Session对象
- `name` Tensor的名称

返回：输出Tensor对象

---
### getSessionInfo
```cpp
bool getSessionInfo(const Session* session, SessionInfoCode code, void* ptr);
```
根据指定类型获取`Session`的信息

参数：
- `session` 要获取信息的Session对象
- `code` 获取的信息类型，使用[SessionInfoCode](Interpreter.html#sessioninfocode)
- `ptr` 将信息存储在该指针中

返回：是否支持获取`code`类型的信息

---
### getSessionOutputAll
```cpp
const std::map<std::string, Tensor*>& getSessionOutputAll(const Session* session) const;
```
返回模型指定会话的所有的输出tensor

参数：
- `session` 持有推理会话数据的Session对象

返回：所有的输出Tensor的名称和指针

---
### getSessionInputAll
```cpp
const std::map<std::string, Tensor*>& getSessionInputAll(const Session* session) const;
```
返回模型指定会话的所有的输入tensor

参数：
- `session` 持有推理会话数据的Session对象

返回：所有的输入Tensor的名称和指针

---
### resizeTensor
```cpp
void resizeTensor(Tensor* tensor, const std::vector<int>& dims);
```
改变tensor形状，并重新分配内存

参数：
- `tensor` 需要改变形状的Tensor对象，一般为输入tensor
- `dims` 新的形状
- `batch,channel,height,width` 新的形状各维度

返回：`void`

---
### getBackend
```cpp
const Backend* getBackend(const Session* session, const Tensor* tensor) const;
```
获取指定`tensor`创建时使用的后端`Backend`；可以在代码中使用该函数来判断当前`Session`的推理实际使用什么后端。

参数：
- `session` tensor相关的会话
- `tensor` 被创建的tensor

返回：创建指定tensor的后端，可能为`nullptr`

---
### bizCode
```cpp
const char* bizCode() const;
```
获取创建该解释器的模型中的`bizCode`

参数：
- `void`

返回：模型中的`bizCode`

---
### uuid
```cpp
const char* uuid() const;
```
获取创建该解释器的模型中的`uuid`

参数：
- `void`

返回：模型中的`uuid`