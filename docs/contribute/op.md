# 自定义算子
## 概述
在添加自定义算子前，请参阅[算子列表](../en/ops)，避免不必要的重复。
### MNN 算子转换与实现结构
MNN 的算子转换与实现如下图，
- 模型转换包括以下步骤，二选一：
   - 训练框架导出的Op与MNN的Op一一对应：前端直接转换
   - 用组合器（参考 tools/converter/source/optimizer/onnxextra 等目录）由 MNN 算子组合。
- MNN 算子实现包括如下步骤
   1. 添加Schema描述（必须）
   2. 添加维度计算（若算子输出维度和输入一致可跳过）
   3. 添加几何计算实现（可选，如果实现几何计算，无须后续在各后端添加算子实现）
   4. 添加各后端算子实现（可选，选择需要部分进行实现）

![image.png](pic1.png)
### 添加算子的流程

![image.png](pic2.png)
简单来说，优先转换，然后组合，然后几何计算，最后各后端实现。

## 添加Schema描述
若添加的算子不在MNN的算子列表中，需要添加模型描述。修改完模型描述后，需要调用generate脚本重新生成模型描述头文件。
### 添加算子类型
在`schema/default/MNN.fbs`文件的OpType列表里追加算子名称，如：
```bash
enum OpType : int {
    AbsVal,
    QuantizedAdd,
    ...
    MyCustomOp
}
```
### 添加算子参数描述
如果算子不包含参数，则可以略过这一步。

首先，在`schema/default/MNN.fbs`文件的OpParameter列表里追加算子参数名称，如：
```bash
union OpParameter {
    QuantizedAdd,
    ArgMax,
    AsString,
    ...
    MyCustomOpParam
}
```
而后，添加参数描述。如果算子来自Caffe，选择`CaffeOps.fbs`；如果算子来自TensorFlow，就使用`TensorflowOp.fbs`。
```bash
table MyCustomOpParam {
    padX:int;
    padY:int;
    kernelX:int;
    kernelY:int;
    strideX:int;
    strideY:int;
    dataType:DataType=DT_FLOAT;
}
```

## 添加模型转换
用户可根据自己使用的框架，选择对应的模型转换模块去添加算子转换的支持。添加完模型转换后，需要重新cmake。

目前，MNN支持TensorFlow、TensorFlow Lite、Caffe、ONNX和TorchScript模型格式的转换。

### TensorFlow模型转换
1. 添加转换类
在`tools/converter/source/tensorflow`下添加`MyCustomOpTf.cpp`。可以直接声明转换类，也可以利用宏定义简化代码。

直接声明示例：
```cpp
class MyCustomOpTf : public tfOpConverter {                                             
    public:                                                                       
        virtual void run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph);
        MyCustomOpTf() {}                                                                   
        virtual ~MyCustomOpTf() {}                                                          
        virtual MNN::OpType opType();                                             
        virtual MNN::OpParameter type();                                          
}
```

等效宏定义示例：
```cpp
DECLARE_OP_CONVERTER(MyCustomOpTf);
```

需要实现`run`、析构、`opType`和`type`函数。其中，`run`函数用于解析模型的proto文件得到参数，然后赋值给flatbuffer自定义参数。参数`srcNode`保存有输入输出节点信息，可以根据输入输出节点在`tempGraph`中找到`TmpNode`。调用函数`find_attr_value(const tensorflow::NodeDef& node, const char* key, tensorflow::AttrValue& value)`获得对应参数的值。

注册转换类：
```cpp
REGISTER_CONVERTER(MyCustomOpTf, MyCustomOp);
```

2. 添加映射
在`OpMapper.hpp`中添加相应的TensorFlow Op名字到MNN Op名字的映射：
```cpp
{"OpName1", MNN::OpType_MyCustomOp},
{"OpName2", MNN::OpType_MyCustomOp},
```

3. 处理Op附带的Const
如果Const不作为此Op的参数，而是看成一个单独的Op，可以忽略此步骤；如果Op要把Const当成参数，要在文件`TmpGraph.cpp`里修改函数`_genMinGraph()`，把相应Const节点的`isCovered`属性设置为true。

### TensorFlow Lite模型转换
1. 添加转换类
在`tools/converter/source/tflite`下添加`MyCustomOpTflite.cpp`。

宏定义示例：
```cpp
DECLARE_OP_COVERTER(MyCustomOpTflite);
```

需要实现函数：
```cpp
MyCustomOpTflite::opType(int quantizedModel);
MyCustomOpTflite::type(int quantizedModel);
MyCustomOpTflite::run(MNN::OpT *dstOp, 
                      const std::unique_ptr<tflite::OperatorT> &tfliteOp, 
                      const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
                      const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
                      const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet,
                      int quantizedModel)
```

其中，`run`函数相比TensorFlow的版本，多一个`quantizedModel`参数。若`quantizedModel`为true，则模型为量化模型，需转为相应的量化Op；若为false，转为浮点Op。在run函数中需要设置输入、输出tensor的index：
```cpp
// set input output index
dstOp->inputIndexes.resize(1);
dstOp->outputIndexes.resize(1);
dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
dstOp->outputIndexes[0] = tfliteOp->outputs[0];
```

注册转换类：
```cpp
using namespace tflite;
REGISTER_CONVERTER(MyCustomOpTflite, BuiltinOperator_OPName);
```

### Caffe模型转换
1. 添加转换类
在`/tools/converter/source/caffe`下添加MyCustomOp.cpp。

类声明示例：
```cpp
class MyCustomOp : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, 
                     const caffe::LayerParameter& parameters, 
                     const caffe::LayerParameter& weight);
    MyCustomOp() {}
    virtual ~MyCustomOp() {}
    virtual MNN::OpType opType();
    virtual MNN::OpParameter type();
};
```

实现`run`、`opType`、`type`函数，在`run`函数中解析caffe参数得到具体参数。其中参数parameters保存有Op的参数信息，weight保存有卷积、BN等数据参数。

注册转换类：
```cpp
static OpConverterRegister<MyCustomOp> a("MyCustomOp");
```

### ONNX模型转换
1. 添加转换类
在`/tools/converter/source/onnx`下添加MyCustomOpOnnx.cpp。

类声明示例：
```cpp
DECLARE_OP_CONVERTER(MyCustomOpOnnx);
```

需要实现函数：
```cpp
MNN::OpType MyCustomOpOnnx::opType();
MNN::OpParameter MyCustomOpOnnx::type();
void MyCustomOpOnnx::run(MNN::OpT* dstOp, 
                         const onnx::NodeProto* onnxNode, 
                         std::vector<const onnx::TensorProto*> initializers);
```
`run`函数中，onnxNode即onnx原始节点信息，权重等数据信息需从initializers取。

注册转换类：
```cpp
REGISTER_CONVERTER(MyCustomOpOnnx, MyCustomOp);
```

## 添加维度计算
如果该Op的输出Tensor大小与第1个输入Tensor一致，并且不需要分析FLOPS，可以跳过这步。添加完形状计算代码后，需要在根目录下运行 python3 tools/scripts/register.py，并重新cmake。

### 添加计算类
在`/source/shape`下添加ShapeMyCustomOp.cpp：
```cpp
class MyCustomOpSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // set tensor->buffer.type
        //                   .dimensions
		//                   .dim[x].extent
        //       			 .dim[x].stride
        //       			 .dim[x].flag
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, 
                                 const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const {
        return flops_for_calc_output_from_input;
    }
};
```

在`onComputeSize`函数中，根据输入tensor的维度信息，计算输出tensor的维度信息，并设置输出tensor的数据类型。计算完成后返回true；若输入维度信息未知返回false。
在`onComputeFlops`函数中，根据输入、输出tensor的维度信息，返回总计算量。

### 注册计算类
```cpp
REGISTER_SHAPE(MyCustomOpSizeComputer, OpType_MyCustomOp);
```

## 添加实现
添加完算子实现后，需要在根目录下运行 python3 tools/scripts/register.py，并重新cmake。

### 添加CPU实现
在`source/backend/CPU`目录下添加`CPUMyCustomOp.hpp`、`CPUMyCustomOp.cpp`。

1. 实现类声明
```c
class CPUMyCustomOp : public Execution {
public:
    // 若执行onExecute需要使用缓存，在此函数中申请，若无可不声明
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, 
                               const std::vector<Tensor *> &outputs) override;
    // 具体的Op执行函数
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, 
                                const std::vector<Tensor *> &outputs) override;
};
```

2. 实现`onResize`和`onExecute`
在`onResize`中，调用`backend()->onAcquireBuffer(&mCache, Backend::DYNAMIC)`进行缓存的申请，调用`backend()->onReleaseBuffer(&mCache, Backend::DYNAMIC)`回收缓存。释放后的内存可以被复用。
在`onExecute`中，做必要的输入的检查，有利于提前发现问题。若执行完毕正确返回NO_ERROR。

3. 注册实现类
```cpp
class CPUMyCustomOpCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, 
                                const std::vector<Tensor *> &outputs, 
                                const MNN::Op *op,
                                Backend *backend) const override {
        return new CPUMyCustomOp(backend);
    }
};
REGISTER_CPU_OP_CREATOR(CPUMyCustomOpCreator, OpType_MyCustomOp);
```

### 添加Metal实现

- 实现类声明

在`source/backend/metal`目录下添加`MetalMyCustomOp.hpp`和`MetalMyCustomOp.cpp`
```cpp
class MetalMyCustomOp : public Execution {
public:
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, 
                               const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
};
```

- 实现`onResize`和`onEncode`

尽量将申请内存和计算group size 的操作放在 onResize 函数中。

onEncode 时，使用传入的 encoder 编排计算任务，不要自行创建 command buffer 或 encoder

- 内存使用

不同于CPU Tensor将数据存储在host指针中，Metal数据指针存放在`deviceId`中，deviceId上存储的是`id<MTLBuffer>`, ，由于内存复用机制，各Tensor有可能共用同一块内存，以offset进行偏移：
```objectivec
auto buffer = (__bridge id<MTLBuffer>)(void *)tensor->deviceId();
auto offset = TensorUtils::getDescribe(tensor)->extra.offset;
```

Metal Op的特定参数等可以通过`id<MTLBuffer>`存储。buffer数据类型可以与tensor不同，buffer甚至可以混合多种数据类型，只需保证创建时指定了正确的长度即可。例如：
```objectivec
auto buffer = [context newDeviceBuffer:2 * sizeof(int) + 2 * sizeof(__fp16) access:CPUWriteOnly];
((__fp16 *)buffer.contents)[0] = mAlpha / mLocalSize;  // alpha
((__fp16 *)buffer.contents)[1] = mBeta;                // beta
((int *)buffer.contents)[1] = mLocalSize;              // local size
((int *)buffer.contents)[2] = inputs[0]->channel();    // channel
```

在创建buffer时，需要指定访问控制权限。目前共有三种权限：

   - `CPUReadWrite`，数据在CPU/GPU间共享存储，一般用于device buffer；
   - `CPUWriteOnly`，数据通过CPU写入后不再读取，一般用于参数buffer；
   - `CPUTransparent`，数据只在GPU中，一般用于heap buffer；

**MNNMetalContext**在创建buffer上，有两套相近的接口，区别只在数据的生命周期上：

   - device占用的内存在单次推理过程中都不会被复用；
   - 而heap占用的内存，在调用`-[MNNMetalContext releaseHeapBuffer:]`之后，可以被其他Op复用；

一般而言，heap只会与**CPUTransparent**一起使用。_heap实际只在iOS 10+上有效，iOS 9-上会回退到device上。_

Metal 内存布局与CPU-FP32-Neon一致，在 Tensor 的 dimentionFormat 为 NC4HW4 时，使用 C4NHW4的排布。否则按默认线性布局。


- 注册实现类

```cpp
class MetalMyCustomOpCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, 
                                const MNN::Op *op, Backend *backend) const {
        return new MetalMyCustomOp(backend);
    }
};
REGISTER_METAL_OP_CREATOR(MetalMyCustomOpCreator, OpType_MyCustomOp);
```

- 工程更新

进入 source/backend/metal 目录，执行 [ python3 MetalCodeGen.py . ] ，更新自注册文件

重新运行一下 CMake ，或者手动在Xcode工程中新加文件

### 添加Vulkan实现
1. 添加Shader
在`source/backend/vulkan/execution/glsl`目录下添加具体的shader(*.comp)。若输入内存布局为`NC4HW4`，则按`image`实现，否则采用buffer实现。可以参考目录下已有实现。然后，执行`makeshader.py`脚本编译Shader。

2. 实现类声明
在目录`source/backend/vulkan/execution/`下添加`VulkanMyCustomOp.hpp`和`VulkanMyCustomOp.cpp`：
```cpp
class VulkanMyCustomOp : public VulkanBasicExecution {
public:
    VulkanMyCustomOp(const Op* op, Backend* bn);
    virtual ~VulkanMyCustomOp();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, 
                       const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;
private:
    // GPU Shader所需的参数
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    // Pipeline
    const VulkanPipeline* mPipeline;
    // Layout Descriptor Set
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
};
```

3. 实现
实现函数`onEncode`，首先需要做内存布局检查：若为`NC4HW4`，则Shader用image实现，否则用buffer。执行完毕返回NO_ERROR。

4. 注册实现类
```cpp
class VulkanMyCustomOpCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, 
                                const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanMyCustomOp(op, backend);
    }
};
static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_MyCustomOp, new VulkanMyCustomOpCreator);
    return true;
}();
```

### 添加OpenCL实现
1. 添加Kernel
在`source/backend/opencl/execution/cl`目录添加具体的kernel(*.cl)。目前feature map均使用`image2d`实现。可以参考目录下已有实现。然后执行`opencl_codegen.py`来生成kernel映射。

2. 实现类声明
在目录`source/backend/opencl/execution/`下添加`MyCustomOp.h`和`MyCustomOp.cpp`：
```cpp
template <typename T>
class MyCustomOp : public Execution {
public:
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, 
                               const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, 
                                const std::vector<Tensor *> &outputs) override;
};
```

3. 实现
实现函数`onResize`(可选)、`onExecute`。执行完毕返回NO_ERROR。

4. 注册实现类
```cpp
OpenCLCreatorRegister<TypedCreator<MyCustomOp<cl_data_t>>> __my_custom_op(OpType_MyCustomOp);
```

### 添加OpenGL实现
1. 添加Shader
在`source/backend/opengl/glsl`下添加具体的shader(*.glsl)，不用加文件头，feature map 均采用`image3d`表示。可以参考目录下已有实现。而后，在`source/backend/opengl`目录下执行`makeshader.py`。

2. 添加Executor
在`source/backend/opengl/execution/`目录下添加`GLMyCustomOp.h`和`GLMyCustomOp.cpp`：
```cpp
class GLMyCustomOp : public Execution {
public:
    GLMyCustomOp(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLMyCustomOp();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, 
                                const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, 
                               const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
};
```

3. 实现
实现函数`onResize`(可选)、`onExecute`。执行完毕返回NO_ERROR。

4. 注册实现类-
```cpp
GLCreatorRegister<TypedCreator<GLMyCustomOp>> __my_custom_op(OpType_MyCustomOp);
```

