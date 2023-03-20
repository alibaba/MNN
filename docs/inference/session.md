# Session API使用

## 创建会话
### 概述
使用MNN推理时，有两个层级的抽象，分别是`解释器Interpreter`和`会话Session`。`Interpreter`是模型数据的持有者；`Session`通过`Interpreter`创建，是推理数据的持有者。多个推理可以共用同一个模型，即，多个`Session`可以共用一个`Interpreter`。

在创建完`Session`，且不再创建`Session`或更新训练模型数据时，`Interpreter`可以通过`releaseModel`函数释放模型数据，以节省内存。


### 创建Interpreter
有两种创建Interpreter的方法：

- 通过磁盘文件创建
```cpp
/**
 * @brief create net from file.
 * @param file  given file.
 * @return created net if success, NULL otherwise.
 */
static Interpreter* createFromFile(const char* file);
```

- 通过内存数据创建
```cpp
/**
 * @brief create net from buffer.
 * @param buffer    given data buffer.
 * @param size      size of data buffer.
 * @return created net if success, NULL otherwise.
 */
static Interpreter* createFromBuffer(const void* buffer, size_t size);
```

**函数返回的Interpreter实例是通过`new`创建的，务必在不再需要时，通过`delete`释放，以免造成内存泄露。**

### 创建Session
一般通过`Interpreter::createSession`创建Session：
```cpp
/**
 * @brief create session with schedule config. created session will be managed in net.
 * @param config session schedule config.
 * @return created session if success, NULL otherwise.
 */
Session* createSession(const ScheduleConfig& config);
```

函数返回的Session实例是由Interpreter管理，随着Interpreter销毁而释放，一般不需要关注。也可以在不再需要时，调用`Interpreter::releaseSession`释放，减少内存占用。

**创建Session 一般而言需要较长耗时，而Session在多次推理过程中可以重复使用，建议只创建一次多次使用。**

#### 简易模式
一般情况下，不需要额外设置调度配置，函数会根据模型结构自动识别出调度路径、输入输出，例如：
```cpp
ScheduleConfig conf;
Session* session = interpreter->createSession(conf);
```

#### 调度配置
调度配置定义如下：
```cpp
/** session schedule config */
struct ScheduleConfig {
    /** which tensor should be kept */
    std::vector<std::string> saveTensors;
    /** forward type */
    MNNForwardType type = MNN_FORWARD_CPU;
    /** CPU:number of threads in parallel , Or GPU: mode setting*/
    union {
        int numThread = 4;
        int mode;
    };

    /** subpath to run */
    struct Path {
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;

        enum Mode {
            /**
             * Op Mode
             * - inputs means the source op, can NOT be empty.
             * - outputs means the sink op, can be empty.
             * The path will start from source op, then flow when encounter the sink op.
             * The sink op will not be compute in this path.
             */
            Op = 0,

            /**
             * Tensor Mode
             * - inputs means the inputs tensors, can NOT be empty.
             * - outputs means the outputs tensors, can NOT be empty.
             * It will find the pipeline that compute outputs from inputs.
             */
            Tensor = 1
        };

        /** running mode */
        Mode mode = Op;
    };
    Path path;

    /** backup backend used to create execution when desinated backend do NOT support any op */
    MNNForwardType backupType = MNN_FORWARD_CPU;

    /** extra backend config */
    BackendConfig* backendConfig = nullptr;
};
```

推理时，主选后端由`type`指定，默认为CPU。若模型中存在主选后端不支持的算子，这些算子会使用由`backupType`指定的备选后端运行。

推理路径包括由`path`的`inputs`到`outputs`途径的所有算子，在不指定时，会根据模型结构自动识别。为了节约内存，MNN会复用`outputs`之外的tensor内存。如果需要保留中间tensor的结果，可以使用`saveTensors`保留tensor结果，避免内存复用。

CPU推理时，并发数与线程数可以由`numThread`修改。`numThread`决定并发数的多少，但具体线程数和并发效率，不完全取决于`numThread`：

- iOS，线程数由系统GCD决定；
- 启用`MNN_USE_THREAD_POOL`时，线程数取决于第一次配置的大于1的`numThread`；
- OpenMP，线程数全局设置，实际线程数取决于最后一次配置的`numThread`；

GPU推理时，可以通过mode来设置GPU运行的一些参量选择(暂时只支持OpenCL)。GPU mode参数如下：
```c
typedef enum {
    // choose one tuning mode Only
    MNN_GPU_TUNING_NONE    = 1 << 0,/* Forbidden tuning, performance not good */
    MNN_GPU_TUNING_HEAVY  = 1 << 1,/* heavily tuning, usually not suggested */
    MNN_GPU_TUNING_WIDE   = 1 << 2,/* widely tuning, performance good. Default */
    MNN_GPU_TUNING_NORMAL = 1 << 3,/* normal tuning, performance may be ok */
    MNN_GPU_TUNING_FAST   = 1 << 4,/* fast tuning, performance may not good */
    
    // choose one opencl memory mode Only
    /* User can try OpenCL_MEMORY_BUFFER and OpenCL_MEMORY_IMAGE both, then choose the better one according to performance*/
    MNN_GPU_MEMORY_BUFFER = 1 << 6,/* User assign mode */
    MNN_GPU_MEMORY_IMAGE  = 1 << 7,/* User assign mode */
} MNNGpuMode;
```
目前支持tuning力度以及GPU memory用户可自由设置。例如：
```c
MNN::ScheduleConfig config;
config.mode = MNN_GPU_TUNING_NORMAL | MNN_GPU_MEMORY_IMAGE;
```
tuning力度选取越高，第一次初始化耗时越多，推理性能越佳。如果介意初始化时间过长，可以选取MNN_GPU_TUNING_FAST或者MNN_GPU_TUNING_NONE，也可以同时通过下面的cache机制，第二次之后就不会慢。GPU_Memory用户可以指定使用MNN_GPU_MEMORY_BUFFER或者MNN_GPU_MEMORY_IMAGE，用户可以选择性能更佳的那一种。如果不设定，框架会采取默认判断帮你选取(不保证一定性能最优)。

**上述CPU的numThread和GPU的mode，采用union联合体方式，共用同一片内存。用户在设置的时候numThread和mode只需要设置一种即可，不要重复设置。**

**对于GPU初始化较慢的问题，提供了Cache机制**。后续可以直接加载cache提升初始化速度。

- 具体可以参考tools/cpp/MNNV2Basic.cpp里面setCacheFile设置cache方法进行使用。
- 当模型推理输入尺寸有有限的多种时，每次resizeSession后调用updateCacheFile更新cache文件。
- 当模型推理输入尺寸无限随机变化时，建议config.mode设为1，关闭MNN_GPU_TUNING。


此外，可以通过`backendConfig`设定后端的额外参数。具体见下。

#### 后端配置
后端配置定义如下：
```c
struct BackendConfig {
    enum MemoryMode {
        Memory_Normal = 0,
        Memory_High,
        Memory_Low
    };
    
    MemoryMode memory = Memory_Normal;
    
    enum PowerMode {
        Power_Normal = 0,
        Power_High,
        Power_Low
    };
    
    PowerMode power = Power_Normal;
    
    enum PrecisionMode {
        Precision_Normal = 0,
        Precision_High,
        Precision_Low,
        Precision_Low_BF16
    };
    
    PrecisionMode precision = Precision_Normal;
    
    /** user defined context */
    void* sharedContext = nullptr;
};
```

`memory`、`power`、`precision`分别为内存、功耗和精度偏好。支持这些选项的后端会在执行时做出相应调整；若不支持，则忽略选项。

示例：
后端 **OpenCL**
**precision 为 Low 时，使用 fp16 存储与计算**，计算结果与CPU计算结果有少量误差，实时性最好；precision 为 Normal 时，使用 fp16存储，计算时将fp16转为fp32计算，计算结果与CPU计算结果相近，实时性也较好；precision 为 High 时，使用 fp32 存储与计算，实时性下降，但与CPU计算结果保持一致。

后端 CPU
**precision 为 Low 时，根据设备情况开启 FP16 计算**
**precision 为 Low_BF16 时，根据设备情况开启 BF16 计算**

`sharedContext`用于自定义后端，用户可以根据自身需要赋值。

### 创建多段路径Session
需要对推理路径做出更为复杂的配置时，可以通过调度配置组来实现：
```cpp
/**
 * @brief create multi-path session with schedule configs. created session will be managed in net.
 * @param configs session schedule configs.
 * @return created session if success, NULL otherwise.
 */
Session* createMultiPathSession(const std::vector<ScheduleConfig>& configs);
```


每个调度配置可以独立配置路径、选项。


### 共享运行时资源
默认情况下，在createSession时对应create单独一个 Runtime。对于串行的一系列模型，可以先单独创建Runtime ，然后在各 Session 创建时传入，使各模型用共享同样的运行时资源（对CPU而言为线程池、内存池，对GPU而言Kernel池等）。

示例:
```cpp
ScheduleConfig config;
config.numberThread = 4;
auto runtimeInfo = Interpreter::createRuntime({config});

/*创建第一个模型*/
std::shared_ptr<Interpreter> net1 = Interpreter::createFromFile("1.mnn");
auto session1 = net1->createSession(config, runtimeInfo);

/*创建第二个模型*/
std::shared_ptr<Interpreter> net2 = Interpreter::createFromFile("2.mnn");
auto session2 = net2->createSession(config, runtimeInfo);

/*创建第三个模型*/
std::shared_ptr<Interpreter> net3 = Interpreter::createFromFile("3.mnn");
auto session3 = net3->createSession(config, runtimeInfo);

// 这样 session1, session2, session3 共用同一个Runtime

/*使用*/
/* 填充输入1..... */
net1->runSession(session1);

/* 读取输出1 填充输入2..... */
net2->runSession(session2);

/* 读取输出2 填充输入3..... */
net3->runSession(session3);
```

## 输入数据
### 获取输入tensor
```cpp
/**
 * @brief get input tensor for given name.
 * @param session   given session.
 * @param name      given name. if NULL, return first input.
 * @return tensor if found, NULL otherwise.
 */
Tensor* getSessionInput(const Session* session, const char* name);

/**
 * @brief get all input tensors.
 * @param session   given session.
 * @return all output tensors mapped with name.
 */
const std::map<std::string, Tensor*>& getSessionInputAll(const Session* session) const;
```

`Interpreter`上提供了两个用于获取输入`Tensor`的方法：`getSessionInput`用于获取单个输入tensor，
`getSessionInputAll`用于获取输入tensor映射。

在只有一个输入tensor时，可以在调用`getSessionInput`时传入NULL以获取tensor。

### 拷贝数据
NCHW示例，适用 ONNX / Caffe / Torchscripts 转换而来的模型：
```cpp
auto inputTensor = interpreter->getSessionInput(session, NULL);
auto nchwTensor = new Tensor(inputTensor, Tensor::CAFFE);
// nchwTensor-host<float>()[x] = ...
inputTensor->copyFromHostTensor(nchwTensor);
delete nchwTensor;
```


NHWC示例，适用于由 Tensorflow / Tflite 转换而来的模型：
```cpp
auto inputTensor = interpreter->getSessionInput(session, NULL);
auto nhwcTensor = new Tensor(inputTensor, Tensor::TENSORFLOW);
// nhwcTensor-host<float>()[x] = ...
inputTensor->copyFromHostTensor(nhwcTensor);
delete nhwcTensor;
```

通过这类拷贝数据的方式，用户只需要关注自己创建的tensor的数据布局，`copyFromHostTensor`会负责处理数据布局上的转换（如需）和后端间的数据拷贝（如需）。


### 直接填充数据
```cpp
auto inputTensor = interpreter->getSessionInput(session, NULL);
inputTensor->host<float>()[0] = 1.f;
```

`Tensor`上最简洁的输入方式是直接利用`host`填充数据，但这种使用方式仅限于CPU后端，其他后端需要通过`deviceid`来输入。另一方面，用户需要自行处理`NC4HW4`、`NHWC`数据格式上的差异。

对于非CPU后端，或不熟悉数据布局的用户，宜使用拷贝数据接口。



### 图像处理
MNN中提供了CV模块，可以帮助用户简化图像的处理，还可以免于引入opencv、libyuv等图片处理库。

1. 支持目标Tensor为float或 uint8_t 的数据格式
2. 支持目标Tensor为NC4HW4或NHWC的维度格式
3. CV模块支持直接输入Device Tensor，也即由Session中获取的Tensor。


#### 图像处理配置
```cpp
struct Config
{
    Filter filterType = NEAREST;
    ImageFormat sourceFormat = RGBA;
    ImageFormat destFormat = RGBA;
    
    //Only valid if the dest type is float
    float mean[4] = {0.0f,0.0f,0.0f, 0.0f};
    float normal[4] = {1.0f, 1.0f, 1.0f, 1.0f};
};
```

`CV::ImageProcess::Config`中

- 通过`sourceFormat`和`destFormat`指定输入和输出的格式，当前支持`RGBA`、`RGB`、`BGR`、`GRAY`、`BGRA`、`YUV_NV21、YUV_NV12`
- 通过`filterType`指定插值的类型，当前支持`NEAREST`、`BILINEAR`和`BICUBIC`三种插值方式
- 通过`mean`和`normal`指定均值归一化，但数据类型不是浮点类型时，设置会被忽略

#### 图像变换矩阵
`CV::Matrix`移植自Android 系统使用的Skia引擎，用法可参考Skia的Matrix：[https://skia.org/user/api/SkMatrix_Reference](https://skia.org/user/api/SkMatrix_Reference)。

需要注意的是，ImageProcess中设置的Matrix是从目标图像到源图像的变换矩阵。使用时，可以按源图像到目标图像的变换设定，最后取逆。例如：
```cpp
// 源图像：1280x720
// 目标图像：逆时针旋转90度再缩小到原来的1/10，即变为72x128

Matrix matrix;
// 重设为单位矩阵
matrix.setIdentity();
// 缩小，变换到 [0,1] 区间：
matrix.postScale(1.0f / 1280, 1.0f / 720);
// 以中心点[0.5, 0.5]旋转90度
matrix.postRotate(90, 0.5f, 0.5f);
// 放大回 72x128
matrix.postScale(72.0f, 128.0f);
// 转变为 目标图像 -> 源图的变换矩阵
matrix.invert(&matrix);
```

#### 图像处理实例
MNN中使用`CV::ImageProcess`进行图像处理。`ImageProcess`内部包含一系列缓存，为了避免内存的重复申请释放，建议将其作缓存，仅创建一次。我们使用`ImageProcess`的`convert`填充tensor数据。
```cpp
/*
 * source: 源图像地址
 * iw: 源图像宽
 * ih：源图像高，
 * stride：源图像对齐后的一行byte数（若不需要对齐，设成 0（相当于 iw*bpp））
 * dest: 目标 tensor，可以为 uint8 或 float 类型
 */
ErrorCode convert(const uint8_t* source, int iw, int ih, int stride, Tensor* dest);
```

#### 完整示例
```cpp
auto input  = net->getSessionInput(session, NULL);
auto output = net->getSessionOutput(session, NULL);

auto dims  = input->shape();
int bpp    = dims[1]; 
int size_h = dims[2];
int size_w = dims[3];

auto inputPatch = argv[2];
FREE_IMAGE_FORMAT f = FreeImage_GetFileType(inputPatch);
FIBITMAP* bitmap = FreeImage_Load(f, inputPatch);
auto newBitmap = FreeImage_ConvertTo32Bits(bitmap);
auto width = FreeImage_GetWidth(newBitmap);
auto height = FreeImage_GetHeight(newBitmap);
FreeImage_Unload(bitmap);

Matrix trans;
//Dst -> [0, 1]
trans.postScale(1.0/size_w, 1.0/size_h);
//Flip Y  （因为 FreeImage 解出来的图像排列是Y方向相反的）
trans.postScale(1.0,-1.0, 0.0, 0.5);
//[0, 1] -> Src
trans.postScale(width, height);

ImageProcess::Config config;
config.filterType = NEAREST;
float mean[3] = {103.94f, 116.78f, 123.68f};
float normals[3] = {0.017f,0.017f,0.017f};
::memcpy(config.mean, mean, sizeof(mean));
::memcpy(config.normal, normals, sizeof(normals));
config.sourceFormat = RGBA;
config.destFormat = BGR;

std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
pretreat->setMatrix(trans);
pretreat->convert((uint8_t*)FreeImage_GetScanLine(newBitmap, 0), width, height, 0, input);
net->runSession(session);
```

### 可变维度
```cpp
/**
 * @brief resize given tensor.
 * @param tensor    given tensor.
 * @param dims      new dims. at most 6 dims.
 */
void resizeTensor(Tensor* tensor, const std::vector<int>& dims);

/**
 * @brief resize given tensor by nchw.
 * @param batch  / N.
 * @param channel   / C.
 * @param height / H.
 * @param width / W
 */
void resizeTensor(Tensor* tensor, int batch, int channel, int height, int width);

/**
 * @brief call this function to get tensors ready. output tensor buffer (host or deviceId) should be retrieved
 *        after resize of any input tensor.
 * @param session given session.
 */
void resizeSession(Session* session);
```
在输入Tensor维度不确定或需要修改时，需要调用`resizeTensor`来更新维度信息。这种情况一般发生在未设置输入维度和输入维度信息可变的情况。更新完所有Tensor的维度信息之后，需要再调用`resizeSession`来进行预推理，进行内存分配及复用。示例如下：
```cpp
auto inputTensor = interpreter->getSessionInput(session, NULL);
interpreter->resizeTensor(inputTensor, {newBatch, newChannel, newHeight, newWidth});
interpreter->resizeSession(session);
inputTensor->copyFromHostTensor(imageTensor);
interpreter->runSession(session);
```

## 运行会话
MNN中，`Interpreter`一共提供了三个接口用于运行`Session`，但一般来说，简易运行就足够满足绝对部分场景。

### 简易运行
```cpp
/**
 * @brief run session.
 * @param session   given session.
 * @return result of running.
 */
ErrorCode runSession(Session* session) const;
```
传入事先创建好的`Session`即可。

函数耗时并不总是等于推理耗时 —— 在CPU下，函数耗时即推理耗时；在其他后端下，函数可能不会同步等待推理完成，例如GPU下，函数耗时为GPU指令提交耗时。


### 回调运行
```cpp
typedef std::function<bool(const std::vector<Tensor*>&, 
                           const std::string& /*opName*/)> TensorCallBack;

/*
 * @brief run session.
 * @param session   given session.
 * @param before    callback before each op. return true to run the op; return false to skip the op.
 * @param after     callback after each op. return true to continue running; return false to interrupt the session.
 * @param sync      synchronously wait for finish of execution or not.
 * @return result of running.
 */
ErrorCode runSessionWithCallBack(const Session* session, 
                                 const TensorCallBack& before, 
                                 const TensorCallBack& end,
                                 bool sync = false) const;
```
相比于简易运行，回调运行额外提供了：

- 每一个op执行前的回调，可以用于跳过Op执行；
- 每一个op执行后的回调，可以用于中断整个推理；
- 同步等待选项，默认关闭；开启时，所有后端均会等待推理完成，即函数耗时等于推理耗时；



### 计算量评估
```cpp
class MNN_PUBLIC OperatorInfo {
    struct Info;

public:
    /** Operator's name*/
    const std::string& name() const;

    /** Operator's type*/
    const std::string& type() const;

    /** Operator's flops, in M*/
    float flops() const;

protected:
    OperatorInfo();
    ~OperatorInfo();
    Info* mContent;
};
typedef std::function<bool(const std::vector<Tensor*>&, const OperatorInfo*)> TensorCallBackWithInfo;

/*
 * @brief run session.
 * @param session   given session.
 * @param before    callback before each op. return true to run the op; return false to skip the op.
 * @param after     callback after each op. return true to continue running; return false to interrupt the session.
 * @param sync      synchronously wait for finish of execution or not.
 * @return result of running.
 */
ErrorCode runSessionWithCallBackInfo(const Session* session, 
                                     const TensorCallBackWithInfo& before,
                                     const TensorCallBackWithInfo& end, 
                                     bool sync = false) const;
```
一般而言，只有在评估计算量时才会用到的接口。相比于回调运行，在回调时，增加了Op类型和计算量信息。

## 获取输出

### 获取输出tensor
```cpp
/**
 * @brief get output tensor for given name.
 * @param session   given session.
 * @param name      given name. if NULL, return first output.
 * @return tensor if found, NULL otherwise.
 */
Tensor* getSessionOutput(const Session* session, const char* name);

/**
 * @brief get all output tensors.
 * @param session   given session.
 * @return all output tensors mapped with name.
 */
const std::map<std::string, Tensor*>& getSessionOutputAll(const Session* session) const;
```

`Interpreter`上提供了两个用于获取输出`Tensor`的方法：`getSessionOutput`用于获取单个输出tensor，
`getSessionOutputAll`用于获取输出tensor映射。

在只有一个输出tensor时，可以在调用`getSessionOutput`时传入NULL以获取tensor。

**注意：当`Session`析构之后使用`getSessionOutput`获取的`Tensor`将不可用**

### 拷贝数据
**不熟悉MNN源码的用户，必须使用这种方式获取输出！！！**
NCHW （适用于 Caffe / TorchScript / Onnx 转换而来的模型）示例：
```cpp
auto outputTensor = interpreter->getSessionOutput(session, NULL);
auto nchwTensor = new Tensor(outputTensor, Tensor::CAFFE);
outputTensor->copyToHostTensor(nchwTensor);
auto score = nchwTensor->host<float>()[0];
auto index = nchwTensor->host<float>()[1];
// ...
delete nchwTensor;
```

NHWC （适用于 Tensorflow / Tflite 转换而来的模型）示例：
```cpp
auto outputTensor = interpreter->getSessionOutput(session, NULL);
auto nhwcTensor = new Tensor(outputTensor, Tensor::TENSORFLOW);
outputTensor->copyToHostTensor(nhwcTensor);
auto score = nhwcTensor->host<float>()[0];
auto index = nhwcTensor->host<float>()[1];
// ...
delete nhwcTensor;
```

**通过这类拷贝数据的方式，用户只需要关注自己创建的tensor的数据布局，`copyToHostTensor`会负责处理数据布局上的转换（如需）和后端间的数据拷贝（如需）。**



### 直接读取数据
**由于绝大多数用户都不熟悉MNN底层数据布局，所以不要使用这种方式！！！**
```cpp
auto outputTensor = interpreter->getSessionOutput(session, NULL);
auto score = outputTensor->host<float>()[0];
auto index = outputTensor->host<float>()[1];
// ...
```

`Tensor`上最简洁的输出方式是直接读取`host`数据，但这种使用方式仅限于CPU后端，其他后端需要通过`deviceid`来读取数据。另一方面，用户需要自行处理`NC4HW4`、`NHWC`数据格式上的差异。

**对于非CPU后端，或不熟悉数据布局的用户，宜使用拷贝数据接口。**

## 示例代码
完整的示例代码可以参考`demo/exec/`文件夹中的以下源码文件：
- `pictureRecognition.cpp` 使用`Session`执行模型推理进行图片分类，使用`ImageProcess`进行前处理
- `multiPose.cpp` 使用`Session`执行模型推理进行姿态检测，使用`ImageProcess`进行前处理
- `segment.cpp` 使用`Session`执行模型推理进行图像分割，使用`ImageProcess`进行前处理，`Expr`进行后处理
- `pictureRotate.cpp` 使用`ImageProcess`进行图像处理
