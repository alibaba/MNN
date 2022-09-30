# Module API使用
## 概念说明
`Module`接口可以用于模型训练与模型推理
- 模型训练时用户可以继承`Module`类增加自己的实现用来训练；
- 模型推理与`Session`的区别是不需要用户显示resize，支持控制流，所以当模型中有`if`或`while`时必须使用`Module`推理
### 相关数据结构
- `Module` Module接口的核心类，表示一个模型的虚类；实际加载模型时会创建其子类
- `Executor` 包含若干个`RuntimeManager`，提供内存管理接口，每个`Executor`必须在单线程环境下运行。默认提供全局 `Executor`，需要并发执行时，可自行创建。
- `ExecutorScope`  用于在子线程中绑定`Executor`，多线程并发必需
- `VARP` 作为`Module`的输入输出，也是[Expr API](expr.md)中的基础数据结构

## 工作流程
创建Executor(可选) -> 创建Module -> 创建输入VARP -> 使用Module::forwad推理 -> 使用输出VARP -> 销毁Module -> 销毁Executor(可选)
### 创建Executor
`Executor`给用户提供接口来配置推理后端、线程数等属性，以及做性能统计、算子执行的回调函数、内存回收等功能。 提供一个全局的Exector对象，用户不用创建或持有对象即可直接使用。
```cpp
// 新建Exector
NNForwardType type = MNN_FORWARD_CPU;
MNN::BackendConfig backend_config;    // default backend config 
std::shared_ptr<MNN::Express::Executor> executor(
    MNN::Express::Executor::newExecutor(type, backend_config, 4));
MNN::Express::ExecutorScope scope(executor);
// 使用默认全局Exector
MNN::BackendConfig backend_config;    // default backend config 
MNN::Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(type, backend_config, 4);
``` 
### 创建Module
`Module`可以通过制定模型，输入输出的名称，配置文件创建；也可以从现有的`Module`对象`clone`
```cpp
// 从模型文件加载并创建新Module
const std::string model_file = "/tmp/mymodule.mnn"; // model file with path
const std::vector<std::string> input_names{"input_1", "input_2", "input_3"};
const std::vector<std::string> output_names{"output_1"};
Module::Config mdconfig; // default module config
std::unique_ptr<Module> module; // module 
module.reset(Module::load(input_names, output_names, model_filename.c_str(), &mdconfig));
// 从现有Module创建新Module，可用于多进程并发
std::unique_ptr<Module> module_shallow_copy;
module_shallow_copy.reset(Module::clone(module.get()));
```
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

**注意：当`Module`析构之后使用`onForward`返回的`VARP`将不可用**

```cpp
std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs);
```

## 使用Module进行模型推理
使用Module进行推理时支持控制流算子，所以对于语音模型常用Module进行推理。示例代码：

```cpp
int dim = 224；
std::vector<VARP> inputs(3);
inputs[0] = _Input({1, dim}, NHWC, halide_type_of<int>());
inputs[1] = _Input({1, dim}, NHWC, halide_type_of<int>());
inputs[2] = _Input({1, dim}, NHWC, halide_type_of<int>());

// 设置输入数据
std::vector<int*> input_pointer = {inputs[0]->writeMap<int>(),
                                   inputs[1]->writeMap<int>(),
                                   inputs[2]->writeMap<int>()};
for (int i = 0; i < inputs[0]->getInfo->size; ++i) {
    input_pointer[0] = i + 1;
    input_pointer[1] = i + 2;
    input_pointer[2] = i + 3;
}
// 执行推理
std::vector<VARP> outputs  = module->onForward(inputs);
// 获取输出
auto output_ptr = outputs[0]->readMap<float>();
```

可以使用回调函数进行调试，与[runSessionWithCallBack](session.html#id19)相似。示例代码：
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

// set callback function
Express::Executor::getGlobalExecutor()->setCallBack(std::move(beforeCallBack), std::move(callBack));

// forward would trigger callback
std::vector<VARP> outputs  = user_module->onForward(inputs);
```

## 示例代码
完整的示例代码可以参考`demo/exec/`文件夹中的以下源码文件：
- `pictureRecognition_module.cpp` 使用`Module`执行图像分类，使用`ImageProcess`进行前处理，`Expr`进行后处理
- `multithread_imgrecog.cpp` 使用`Module`多线程并发执行图像分类，使用`ImageProcess`进行前处理，`Expr`进行后处理
- `transformerDemo.cpp` 使用`Module`执行Transformer模型推理