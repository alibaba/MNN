# Module API使用
## 概念说明
`Module`接口可以用于模型训练与模型推理
- 模型训练时用户可以继承`Module`类增加自己的实现用来训练；
- 模型推理与`Session`的区别是不需要用户显式resize，支持控制流，所以当模型中有`if`或`while`时必须使用`Module`推理
### 相关数据结构
- `Module` Module接口的核心类，表示一个模型的虚类；实际加载模型时会创建其子类
- `Executor` 提供内存管理和后端资源管理能力，每个`Executor`必须在单线程环境下运行。同一个`Executor`可以用于多个顺序执行的`Module`
- `ExecutorScope`  用于在子线程中绑定`Executor`，多线程并发必需。默认在创建`Module`时使用全局 `Executor`，如果有多个Module在不同线程并发执行时，需要各自创建`Executor`，并用`ExecutorScope`绑定。
- `VARP` 是`Module`的输入输出，也是[Expr API](expr.md)中的基础数据结构

## 工作流程
创建和配置Executor -> 创建 RuntimeManager(可选) -> 创建Module -> 创建输入VARP -> 使用Module::forwad推理 -> 使用输出VARP -> 销毁Module -> 销毁Executor
### 创建和配置Executor
`Executor`给用户提供接口来配置推理后端、线程数等属性，以及做性能统计、算子执行的回调函数、内存回收等功能。 推荐针对自身模块创建单独的Executor ，若使用全局的Exector对象，对于多个模块在不同线程运行时可能会发生冲突。
```cpp
// 创建Exector
MNN::BackendConfig backendConfig;    // default backend config 
std::shared_ptr<MNN::Express::Executor> executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);

// 设置使用4线程+CPU
executor->setGlobalExecutorConfig(MNN_FORWARD_CPU, backend_config, 4);

// 绑定Executor，在创建/销毁/使用Module或进行表达式计算之前都需要绑定
MNN::Express::ExecutorScope _s(executor);

``` 

### （可选）创建 RuntimeManager
Executor 的配置会同时影响Module和表达式计算的后端配置。

*** 如下示例会触发表达式计算，若 Executor 设置为 OPENCL ，则该计算会用OpenCL后端实现
```cpp
MNN::Express::VARP X;
MNN::Express::VARP Y = MNN::Express::_Sign(X);
float* yPtr = Y->readMap<float>();
```

若希望仅在该Module中采用某种后端配置（比如Module使用GPU但表达式计算使用CPU），可额外创建 RuntimeManager ，并在创建 Module 时传入
```cpp
MNN::ScheduleConfig sConfig;
sConfig.type = MNN_FORWARD_OPENCL;

std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(sConfig), MNN::Express::Executor::RuntimeManager::destroy);
rtmgr->setCache(".cachefile");
```

RuntimeManager 可以设置 hint , mode , cache, externalpath ，以支持扩展功能。

```
void setCache(std::string cacheName);
void updateCache();
void setMode(Interpreter::SessionMode mode);
void setHint(Interpreter::HintMode mode, int value);
void setExternalPath(std::string path, int type);
bool getInfo(Interpreter::SessionInfoCode code, void* ptr);
```

#### cache 设置
对于GPU后端（Metal/OpenCL等），可以设置缓存文件路径，存储AutoTuning结果和Program编译结果，以加速第二次之后的Module load 过程。

```
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setCache(cacheFileName);

    std::shared_ptr<Module> module(Module::load(inputNames, outputNames, modelName.c_str(), rtmgr, mdConfig));
    /*... Make Inputs*/
    auto outputs = module->onForward(inputs);

    // Update cache file
    rtmgr->updateCache();
```

#### mode 设置
可以通过设置mode开启/关闭一些功能，示例：

```
// 创建出来的 Module 支持插入回调函数
rtmgr->setMode(Interpreter::Session_Debug);
```

并非所有枚举都适用 Module 的创建，有效值如下：

- Interpreter::SessionMode::Session_Debug : 支持逐算子调试
- Interpreter::SessionMode::Session_Release : 关闭逐算子调试功能，可以轻微提升性能【默认选项】
- Interpreter::SessionMode::Session_Backend_Fix : 固定使用用户设置的后端【默认选项】
- Interpreter::SessionMode::Session_Backend_Auto : MNN根据用户倾向，预估load Module耗时，如果耗时较短则使用用户设置的后端，否则使用CPU


#### hint 设置
通过 hint 设置，可以在后端支持的情况下设置相应属性，有效值如下：

- Interpreter::HintMode::WINOGRAD_MEMORY_LEVEL ：使用 Winograd 算法优化卷积时，内存占用倾向，默认为 3 ，若希望降低内存占用可设为 0 
- Interpreter::HintMode::GEOMETRY_COMPUTE_MASK ：几何计算相关优化开关，1为区域合并，2为复合区域合并，4为使用loop算子，8为支持几何计算重计算，需要多个功能开启时把对应值叠加。默认为功能全开。
- Interpreter::HintMode::CPU_LITTLECORE_DECREASE_RATE ：对于 Android 设备存在大中小核的情况，大核算力到中核算力的衰减比例。默认为50（中核算力为大核的50%）


#### ExternalPath
在设备可能出现内存不足时，可以通过 setExternalPath 指定路径，让MNN把部分内存用mmap分配。这样操作系统可在内存不足时会将其转换为读写文件，避免内存不足程序闪退。示例：

```
runtime_manager_->setExternalPath("tmp", MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
runtime_manager_->setExternalPath("tmp", MNN::Interpreter::EXTERNAL_FEATUREMAP_DIR);
```

- MNN::Interpreter::EXTERNAL_WEIGHT_DIR : 权重重排后的内存转换为文件存储
- MNN::Interpreter::EXTERNAL_FEATUREMAP_DIR : 中间内存转换为文件存储

### 创建Module
`Module`可以通过指定模型，输入输出的名称，配置文件创建
```cpp
// 从模型文件加载并创建新Module
const std::string model_file = "/tmp/mymodule.mnn"; // model file with path

// 输入名：多个输入时按顺序填入，其顺序与后续 onForward 中的输入数组需要保持一致
const std::vector<std::string> input_names{"input_1", "input_2", "input_3"};

// 输出名，多个输出按顺序填入，其顺序决定 onForward 的输出数组顺序
const std::vector<std::string> output_names{"output_1"};

Module::Config mdconfig; // default module config
std::unique_ptr<Module> module; // module 
// 若 rtMgr 为 nullptr ，Module 会使用Executor的后端配置
module.reset(Module::load(input_names, output_names, model_filename.c_str(), rtMgr, &mdconfig));
```

输入输出的名字可以为空，此时，MNN 会检索模型中的输入/输出填入，在多输入/输出情况下无法保证顺序，需要通过 getInfo 接口查看。

### Module::Config 
创建`Module`时可传入`Module::Config`，具体结构如下：

```cpp
struct Config {
    // Load module as dynamic, default static
    bool dynamic = false;

    // for static mode, if the shape is mutable, set true, otherwise set false to avoid resizeSession freqencily
    bool shapeMutable = true;
    // Pre-rearrange weights or not. Disabled by default.
    // The weights will be rearranged in a general way, so the best implementation
    // may not be adopted if `rearrange` is enabled.
    bool rearrange = false;

    BackendInfo* backend = nullptr;
};
```

#### dynamic
- 默认为 false ，输出的变量为const ，只能得到数据
- 若 dynamic = true ，加载出的模型将按动态图方式运行，会增加额外构图耗时，但可以保存输出变量的计算路径，存成模型
- 若 dynamic = true ，后面的 shapeMutable / rearrange 不再生效

#### shapeMutable
- 默认为 true ，表示输入形状易变，将延迟进行形状相关计算
- 设置为 false 时，会提前申请内存，在 onForward 时做输入数据的拷贝而不是直接使用指针

#### rearrange
- 若为 true ，在创建 Module 时会预先创建卷积算子，做权重重排，以降低运行时的内存
- 目前只支持 CPU 和 CUDA 后端

#### backend
已经废弃，不要设置此项

### 获取模型信息
调用`getInfo`函数可获取`Module`信息，可以参考代码：`tools/cpp/GetMNNInfo.cpp`，[工具](../tools/test.html#getmnninfo)
```cpp
struct Info {
    // Input info load from model
    std::vector<Variable::Info> inputs;
    // The Module's defaultFormat, NCHW or NHWC
    Dimensionformat defaultFormat;
    // Runtime Info
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runTimeManager;
    // Input Names By Order
    std::vector<std::string> inputNames;
    // Output Names By Order
    std::vector<std::string> outputNames;
    // Version of MNN which build the model
    std::string version;
};
const Info* getInfo() const;
```

### 执行推理
调用`onForward`执行推理。

```cpp
std::vector<MNN::Express::VARP> onForward(const std::vector<MNN::Express::VARP>& inputs);
```

示例代码：

```cpp
int dim = 224；
std::vector<VARP> inputs(3);
// 对于 tensoflow 转换过来的模型用 NHWC ，由 onnx 转换过来的模型用 NCHW
inputs[0] = MNN::Express::_Input({1, dim}, NHWC, halide_type_of<int>());
inputs[1] = MNN::Express::_Input({1, dim}, NHWC, halide_type_of<int>());
inputs[2] = MNN::Express::_Input({1, dim}, NHWC, halide_type_of<int>());

// 设置输入数据
std::vector<int*> input_pointer = {inputs[0]->writeMap<int>(),
                                   inputs[1]->writeMap<int>(),
                                   inputs[2]->writeMap<int>()};
for (int i = 0; i < dim; ++i) {
    input_pointer[0] = i + 1;
    input_pointer[1] = i + 2;
    input_pointer[2] = i + 3;
}
// 执行推理
std::vector<MNN::Express::VARP> outputs  = module->onForward(inputs);
// 获取输出
auto output_ptr = outputs[0]->readMap<float>();
```

## 多实例推理

Module API 支持同个模型创建多个实例，分发到不同线程推理。具体步骤如下：

- 【启动】主线程创建基准Module: 配置Executor(可选) -> 创建 RuntimeManager(可选) -> 创建Module
- 【启动】创建子线程，在子线程中创建 Executor 
- 【启动】子线程绑定该线程的Executor ， Clone Module
- 【使用】子线程绑定该线程的executor，使用 Clone 出来的 Module进行推理：创建输入VARP -> 使用Module::forwad推理 -> 使用输出VARP
- 【结束】子线程绑定该线程的executor，销毁 Module
- 【结束】子线程销毁 Executor ，销毁子线程
- 【结束】主线程销毁 Module

### 创建基准Module
第一个实例的创建过程不需要变更

### 每个实例新建Exector
```cpp
NNForwardType type = MNN_FORWARD_CPU;
MNN::BackendConfig backend_config;    // default backend config 
std::shared_ptr<MNN::Express::Executor> executor(
    MNN::Express::Executor::newExecutor(type, backend_config, 1));
```

** 若一个算法流程中有多个模型运行，每份实例单独建一个 Executor 即可。

### 每个实例克隆基准Module

```cpp
// 绑定这个实例的executor，这样不会与其他实例产生内存冲突
MNN::Express::ExecutorScope scope(executor);
std::unique_ptr<MNN::Express::Module> module_thread(MNN::Express::Module::clone(module.get()), MNN::Express::Module::destroy);
```

克隆出来的 Module 与基准 Module 共享不变的权重与常量数据，可以较大地降低新增实例若需的内存。


### 每个实例推理
```cpp
// 每个实例推理之前用 ExecutorScope 绑定这个实例的 executor
MNN::Express::ExecutorScope scope(executor);
std::vector<VARP> inputs;
/* 构建输入......*/
// 执行推理
std::vector<MNN::Express::VARP> outputs = module_thread->onForward(inputs);
/* 使用输出......*/
``` 

### 销毁
```cpp
//每个实例销毁Module之前，也需要用 ExecutorScope 绑定这个实例的 executor
MNN::Express::ExecutorScope scope(executor);
module_thread.reset();
```

## 多线程
Module 的创建与运行依赖其所绑定的 Executor ，若不指定，则为全局 Executor ，并非线程安全。在多线程创建 Module 或进行推理时，会竞争全局 Executor 的资源，需要上锁或绑定不同的 Executor 。

## 调试

Module API 也支持使用回调函数进行调试，与[runSessionWithCallBack](session.html#id19)相似。示例代码：
```cpp
MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const OperatorInfo* info) {

    // do any thing you want.
    auto opName = info->name();
    for (int i = 0; i < ntensors.size(); ++i) {
        auto ntensor    = ntensors[i];
        print("input op name:%s, shape:", opName.c_str());
        ntensor->printShape();
    }
    return true;
};
MNN::TensorCallBackWithInfo callBack = [&](const std::vector<MNN::Tensor*>& ntensors,  const OperatorInfo* info) {
    auto opName = info->name();
    for (int i = 0; i < ntensors.size(); ++i) {
        auto ntensor    = ntensors[i];
        print("output op name:%s, shape:", opName.c_str());
        ntensor->printShape();
    }
    return true;
};

// 设置回调函数，需要时创建该 Module 时的 executor ，非多实例情况下用全局 executor 即可：
Express::Executor::getGlobalExecutor()->setCallBack(std::move(beforeCallBack), std::move(callBack));

// forward would trigger callback
std::vector<VARP> outputs  = user_module->onForward(inputs);
```

## 预推理分离模式
对于满足 Interpreter-Session 运行条件的模型，若用户希望分离预推理（形状计算，几何计算，资源申请，策略搜索）与推理（内容计算）过程，可以设置预推理分离模式。示例代码如下：

```cpp
std::shared_ptr<Module> net(Module::load({"x"}, {"y"}, (const uint8_t*)buffer.data(), buffer.size()), Module::destroy);
// 预推理分离模式
auto code = net->traceOrOptimize(Interpreter::Module_Forward_Seperate);
if (0 != code) {
    // 若模型不支持预推理分离，需要还原设定
    net->traceOrOptimize(Interpreter::Module_Forward_Combine);
}

/*预推理开始*/
x = _Input({1, 3, 2, 2}, NCHW, halide_type_of<int>());
auto input = x->writeMap<int>();
y = net->onForward({x})[0];
auto output = y->readMap<int>();

/*预推理结束，获取输入和输出的数据指针*/

/*内容计算*/
/*
Fill input
*/

// 输入空数组，表示仅进行推理
net1->onForward({});

/*
Use output
*/

```

## 示例代码
完整的示例代码可以参考`demo/exec/`文件夹中的以下源码文件：
- `pictureRecognition_module.cpp` 使用`Module`执行图像分类，使用`ImageProcess`进行前处理，`Expr`进行后处理
- `pictureRecognition_batch.cpp` 使用`Module`执行图像分类，使用`ImageProcess`进行前处理，`Expr`进行后处理
- `multithread_imgrecog.cpp` 使用`Module`多线程并发执行图像分类，使用`ImageProcess`进行前处理，`Expr`进行后处理
- `transformerDemo.cpp` 使用`Module`执行Transformer模型推理
