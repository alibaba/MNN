[English Version](AddOp_EN.md)

# 自定义Op

在自定义Op前，请参阅[Op列表文档](OpList.md)，避免不必要的重复。

在MNN中，添加Op包含如下步骤：
1. [添加Op的模型描述](#添加Op的模型描述)
2. [添加Op的模型转换](#添加Op的模型转换)
3. [计算Op输出tensor大小](#计算Op输出tensor大小)
4. 添加对应的Backend的Op实现，即Execution([CPU](#CPU)|[Metal](#Metal)|[Vulkan](#Vulkan)|[OpenCL](#OpenCL)|[OpenGL](#OpenGL))

## 添加Op的模型描述

首先在[模型参数描述文件夹下](../schema/default)添加相应的Op名字以及参数描述，具体如下:
1. 若此Op来自于Caffe，则在[CaffeOp\.fbs](../schema/default/CaffeOp.fbs)下添加参数描述；若Op来自Tensorflow，则在[TensorflowOp\.fbs](../schema/default/TensorflowOp.fbs)下添加，例如:

```bash
table Pool {
    padX:int;
    padY:int;
    isGlobal:bool=false;
    kernelX:int;
    kernelY:int;
    strideX:int;
    strideY:int;
    type:PoolType;
    padType:PoolPadType;
    dataType:DataType=DT_FLOAT;
}
```
2. 在[MNN\.fbs](../schema/default/MNN.fbs)的OpType列表里添加Op的名字，而后，在OpParameter列表里添加参数描述名字（即table后的Pool）
> 若Op无参数则不需上述步骤1，只需在OpType列表里添加Op名字即可。

## 添加Op的模型转换

> 目前支持Tensorflow、Caffe、ONNX、Tensorflow Lite模型格式的转换。

### Tensorflow模型转换
1. 在[tensorflow目录](../tools/converter/source/tensorflow)下添加xxxTf.cpp
实现可参考[PoolingTf\.cpp](../tools/converter/source/tensorflow/PoolingTf.cpp)
```c++
// 声明具体Op转换PoolingTf, 继承Op转换基类tfOpConverter
// 两种方法
class PoolingTf : public tfOpConverter {                                             
    public:                                                                       
        virtual void run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph);
        PoolingTf() {}                                                                   
        virtual ~PoolingTf() {}                                                          
        virtual MNN::OpType opType();                                             
        virtual MNN::OpParameter type();                                          
}
// 或宏定义
// DECLARE_OP_CONVERTER(PoolingTf);
```
> run()函数用于解析模型的proto文件得到参数，然后赋值给flatbuffer自定义参数：可以调用函数`find_attr_value(const tensorflow::NodeDef& node, const char* key, tensorflow::AttrValue& value)`获得对应参数的值。

> virtual void run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph)，其中参数srcNode保存有输入/输出节点信息，可以根据输入输出节点在tempGraph中找到TmpNode
2. 在OpMapper.hpp中添加相应的tensorflow Op名字到MNN Op名字的映射
```c++
{"MaxPool", MNN::OpType_Pooling},
{"AvgPool", MNN::OpType_Pooling},
```
3. Op附带Const的情况
- 如果Const不作为此Op的参数，而是看成一个单独的Op，可以忽略此步骤
- 如果此Op要把其附带的Const当成参数，要在文件`TmpGraph.cpp`里修改函数`_genMinGraph()`，把相应Const节点的isCovered属性设置为true

### Caffe模型转换
1. 在[caffe目录下](../tools/converter/source/caffe)添加xxx.cpp
具体参考[Pool\.cpp](../tools/converter/source/caffe/Pool.cpp)
在run函数中解析caffe参数得到具体参数
> virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight)，其中参数parameters保存Op的参数信息，weight保存卷积/BN等数据参数

### ONNX模型转换
1. 添加具体Op转换xxxOnnx.cpp，实现如下三个函数
```c++
MNN::OpType PoolingOnnx::opType() { return MNN::OpType_Pooling; }
MNN::OpParameter PoolingOnnx::type() { return MNN::OpParameter_Pool; }

void PoolingOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      std::vector<const onnx::TensorProto*> initializers)
```
> PoolingOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, std::vector<const onnx::TensorProto*> initializers)，其中onnxNode即onnx原始节点信息，权重等数据信息需从initializers取。

### Tensorflow Lite模型转换
1. 添加XXXTflite.cpp
```c++
DECLARE_OP_COVERTER(XXXTflite);

// 需要实现如下函数
XXXTflite::opType(bool quantizedModel);
XXXTflite::type(bool quantizedModel);
XXXTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp, 
   const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
   const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
   const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet,
   bool quantizedModel)
// 接口函数相比tensorflow多一个quantizedModel参数，若quantizedModel为true，
// 则模型为量化模型，需转为相应的量化Op，若为false，转为float Op

// 在run()函数中需要设置输入/输出tensor的index
// set input output index
dstOp->inputIndexes.resize(1);
dstOp->outputIndexes.resize(1);
dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
dstOp->outputIndexes[0] = tfliteOp->outputs[0];

// 注册Op转换
using namespace tflite;
REGISTER_CONVERTER(SqueezeTflite, BuiltinOperator_SQUEEZE);
```

### 计算Op输出tensor大小

1. 根据输入tensor的维度信息，计算输出tensor的维度信息，并设置输出tensor的数据类型。
继承基类SizeComputer，实现`onComputeSize`函数，若输入维度信息未知返回false，计算完成后返回true。例如Pooling：

```c++
class PoolSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto input  = inputs[0];
        auto output = outputs[0];

        ::memcpy(output->buffer().dim, input->buffer().dim, input->buffer().dimensions * sizeof(halide_dimension_t));
        // Pool 参数信息
        auto layer = op->main_as_Pool();
        int outw   = 1;
        int outh   = 1;
        if (!layer->isGlobal()) {
            int w = input->width();
            int h = input->height();
            if (layer->pad_x() > 0)
                w += layer->pad_x() * 2;
            if (layer->pad_y() > 0)
                h += layer->pad_y() * 2;

            // Tensorflow padding mode SAME
            if (layer->pad_type() == PoolPadType_SAME) {
                outw = ceil((float)w / (float)layer->stride_x());
                outh = ceil((float)h / (float)layer->stride_y());
            }
            // Tensorflow padding mode VALID
            else if (layer->pad_type() == PoolPadType_VALID) {
                outw = ceil((float)(w - layer->kernel_x() + 1) / (float)layer->stride_x());
                outh = ceil((float)(h - layer->kernel_y() + 1) / (float)layer->stride_y());
            } 
            else {
                outw = UP_DIV(w - layer->kernel_x(), layer->stride_x()) + 1;
                outh = UP_DIV(h - layer->kernel_y(), layer->stride_y()) + 1;
            }
        }
        // 输入信息未知返回false
        if (outw <= 0 || outh <= 0) {
            return false;
        }
        // Pooling只改变Height，Width
        output->buffer().dim[3].extent = outw;
        output->buffer().dim[2].extent = outh;

        return true;
    }
};

// 注册Shape计算功能
REGISTER_SHAPE(XXXComputer, OpType_XXX);
```

## 添加对应Backend的Execution

### CPU
在[CPU目录下](../source/backend/CPU)添加CPUXXX.hpp、CPUXXX.cpp。

1. 类声明
继承基类Execution，主要实现`onResize()`和`onExecute()`

```c++
class CPUPool : public Execution {
public:
    CPUPool(Backend *b, const Pool *parameter);
    virtual ~CPUPool() = default;

    // 若执行onExecute需要使用缓存，在此函数中申请，若无可不声明
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    // 具体的Op执行函数
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Pool *mParameter;

    Tensor mCacheLine;
};
```

2. 实现
在onResize()中调用`backend()->onAcquireBuffer(&mCacheLine, Backend::DYNAMIC)`进行缓存的申请以及调用`backend()->onReleaseBuffer(&mCacheLine, Backend::DYNAMIC)`回收缓存，便于内存的复用。
在onExecute()需做必要的输入的检查，便于提前发现问题，执行完毕正确返回NO_ERROR。

3. 实现CPU Op Creator，完成注册

```c++
class CPUPoolCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                Backend *backend) const override {
        return new CPUPool(backend, op->main_as_Pool());
    }
};
REGISTER_CPU_Op_CREATOR(CPUPoolCreator, OpType_Pooling);
```

### Metal

在[Metal目录下](../source/backend/Metal)添加MetalXXX.hpp和MetalXXX.cpp。

#### 1. 声明

继承基类Execution，声明构造、析构、`onResize`和`onExecute`函数：

```c++
class MetalPooling : public Execution {
public:
    MetalPooling(Backend *backend, const Pool *pooling);
    virtual ~MetalPooling();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool mGlobal;
    PoolType mPoolType;
    int mKernelX;
    int mKernelY;
    int mStrideX;
    int mStrideY;
    int mPadX;
    int mPadY;
    id<MTLBuffer> mConstBuffer;
};
```

#### 2. 实现
* 不同于CPU Tensor将数据存储在host指针中，Metal数据指针存放在deviceId中，deviceId上存储的是id<MTLBuffer>：

  ```c++
  auto buffer = (__bridge id<MTLBuffer>)(void *)tensor->deviceId();
  ```

* Metal Op的特定参数等可以通过id<MTLBuffer>存储。buffer数据类型可以与tensor不同，buffer甚至可以混合多种数据类型，只需保证创建时指定了正确的长度即可。例如：

  ```c++
  auto buffer = [context newDeviceBuffer:2 * sizeof(int) + 2 * sizeof(__fp16) access:CPUWriteOnly];
  ((__fp16 *)buffer.contents)[0] = mAlpha / mLocalSize;  // alpha
  ((__fp16 *)buffer.contents)[1] = mBeta;                // beta
  ((int *)buffer.contents)[1] = mLocalSize;              // local size
  ((int *)buffer.contents)[2] = inputs[0]->channel();    // channel
  ```

* 在创建buffer时，需要指定访问控制权限。目前共有三种权限：
  CPUReadWrite，数据在CPU/GPU间共享存储，一般用于device buffer；
  CPUWriteOnly，数据通过CPU写入后不再读取，一般用于参数buffer；
  CPUTransparent，数据只在GPU中，一般用于heap buffer。

* **MNNMetalContext**在创建buffer上，有两套相近的接口，区别只在数据的生命周期上：device占用的内存在单次推理过程中都不会被复用；而heap占用的内存，在调用`-[MNNMetalContext releaseHeapBuffer:]`之后，可以被其他Op复用。一般而言，heap只会与**CPUTransparent**一起使用。

  *heap实际只在iOS 10+上有效，iOS 9-上会回退到device上。*

* 使用Metal时，**如非特殊情况，禁止自行创建device和library**。加载library、编译function都是耗时行为，**MNNMetalContext**上做了必要的缓存优化。通过context执行Metal的示例如下：

  ```c++
  auto context   = (__bridge MNNMetalContext *)backend->context();
  auto kernel    = /* metal kernel name NSString */;
  auto encoder   = [context encoder];
  auto bandwidth = [context load:kernel encoder:encoder];
  /* encoder set buffer(s)/sampler(s) */
  [context dispatchEncoder:encoder 
  			           threads:{x, y, z}
        maxThreadsPerGroup:maxThreadsPerThreadgroup]; // recommended way to dispatch
  [encoder endEncoding];
  ```

#### 3. 实现Metal Op Creator，完成注册：

   ```c++
   class MetalPoolingCreator : public MetalBackend::Creator {
   public:
       virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
           return new MetalPooling(backend, op->main_as_Pool());
       }
   };
   static MetalCreatorRegister<MetalPoolingCreator> __ec(OpType_Pooling);
   ```

### Vulkan
#### 1. Shader
在[glsl目录下](../source/backend/vulkan/execution/glsl)添加具体的shader(\*.comp)，Pooling输入内存布局默认为NC4HW4，故按image实现，否则采用buffer实现。

```c++
#version 440 core
layout(std140) buffer;
layout(set=0, binding=0, rgba16f) writeonly restrict uniform image3D uOutput;
layout(set=0, binding=1) uniform sampler3D uInput;

layout(set=0, binding=2) uniform constBuffer {
    ivec4 inputSize;
    ivec4 outputSize;
    ivec2 pad;
    ivec2 kernelSize;
    ivec2 stride;
} uConstant;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 outputSize = uConstant.outputSize.xyz;
    ivec2 spos = pos.xy*uConstant.stride-uConstant.pad;

    if (all(lessThan(pos, outputSize)))
    {
        ivec2 inputSizeXY = uConstant.inputSize.xy;
        vec4 color = vec4(-1000000.0);
        ivec2 sfxy = max(ivec2(0), -spos);
        ivec2 efxy = min(uConstant.kernelSize, inputSizeXY-spos);

        for (int fy=sfxy.y; fy<efxy.y; ++fy)
        {
            for (int fx=sfxy.x; fx<efxy.x; ++fx)
            {
                ivec2 spos_ = spos + ivec2(fx, fy);
                color = max(texelFetch(uInput, ivec3(spos_.x, spos_.y, pos.z), 0), color);
            }
        }
        imageStore(uOutput, pos, color);
    }
}
```

然后执行`makeshader.py`脚本编译Shader。

#### 2. 类声明
在目录[execution](../source/backend/vulkan/execution/)下添加VulkanXXX.hpp和VulkanXXX.cpp。类声明继承类VulkanBasicExecution：

```c++
class VulkanPool : public VulkanBasicExecution {
public:
    VulkanPool(const Op* op, Backend* bn);
    virtual ~VulkanPool();

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;
private:
    // GPU Shader所需的参数
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    // Pipeline
    const VulkanPipeline* mPoolPipeline;
    const Pool* mCommon;
    // Layout Descriptor Set
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
};
```

#### 3. 实现
实现函数onEncode()，首先需要做内存布局检查(若为NC4HW4，则Shader用image实现，否则用buffer)，执行完毕返回NO_ERROR。

#### 4. 实现Vulkan Execution Creator，完成注册
```c++
class VulkanPoolCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanPool(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Pooling, new VulkanPoolCreator);
    return true;
}();
```

### OpenCL
#### 1. Kernel
在[cl目录](../source/backend/opencl/execution/cl)添加具体的kernel(\*.cl)，Pooling按image实现，内存排序为`（ H : batch * height, W : channel/4 * width * channel4）`。

```c++
__kernel void pooling(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __private const int in_height,
                      __private const int in_width, __private const int out_height, __private const int pad_top,
                      __private const int pad_left, __private const int stride_h, __private const int stride_w,
                      __private const int pooling_size_h, __private const int pooling_size_w,
                      __write_only image2d_t output) {
    const int out_channel_idx = get_global_id(0);
    const int out_width_idx   = get_global_id(1);
    const int out_hb_idx      = get_global_id(2);

    if (out_channel_idx >= global_size_dim0 || out_width_idx >= global_size_dim1 || out_hb_idx >= global_size_dim2) {
        return;
    }
    const int out_width = global_size_dim1;

    const int n_b               = out_hb_idx / out_height;
    const int mod_b             = out_hb_idx - mul24(n_b, out_height);
    const int batch_idx         = mul24(n_b, in_height);
    const int in_height_start   = mad24(mod_b, stride_h, -pad_top);
    const int in_width_start    = mad24(out_width_idx, stride_w, -pad_left);
    const int in_channel_offset = mul24(out_channel_idx, in_width);
    DATA_TYPE4 res = (DATA_TYPE4)(MIN_VALUE);
    for (int height = 0; height < pooling_size_h; ++height) {
        int in_height_idx = in_height_start + height;
        in_height_idx     = select(batch_idx + in_height_idx, -1, (in_height_idx < 0 || in_height_idx >= in_height));
        if (in_height_idx != -1) {
            for (int width = 0; width < pooling_size_w; ++width) {
                int in_width_idx = in_width_start + width;
                in_width_idx =
                    select(in_channel_offset + in_width_idx, -1, (in_width_idx < 0 || in_width_idx >= in_width));

                if (in_width_idx != -1) {
                    DATA_TYPE4 in = READ_IMAGE(input, SAMPLER, (int2)(in_width_idx, in_height_idx));
                    res           = MNN_MAX(res, in);
                }
            }
        }
    }
    const int pos = mad24(out_channel_idx, out_width, out_width_idx);
    WRITE_IMAGE(output, (int2)(pos, out_hb_idx), res);
}

```
Then execute `opencl_codegen.py` to generate the string map corresponding to the kernel.

Note: Macro description in kernel

a. `GLOBAL_SIZE_3_DIMS` ：Corresponds to the specified global work group size.
b. `READ_IMAGE / WRITE_IMAGE` ：Read and write pictures.
c. `DATA_TYPE `：The specified data type (float/half/int32).

#### 2. Class declaration

在目录[execution](../source/backend/opencl/execution/)下添加xxx.h和XXX.cpp。类声明继承类Execution，如下

```c++
template <typename T>
class PoolOp : public Execution {
public:
    PoolOp(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~PoolOp() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    bool buildPoolingKernel();
    std::vector<uint32_t> poolLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

private:
    const Pool *mPoolParams;
    PoolType mPoolType;
    PoolPadType mPadType;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mKernels{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    std::vector<int> mInputShape;
    OpenCLBackend *mOpenCLBackend;
};
```
#### 3. 实现
实现函数`onResize( )(可选)`，`onExecute( )`，执行完毕返回NO_ERROR。

#### 4. 实现OpenCL Op Creator以及注册

如下

```c++
OpenCLCreatorRegister<TypedCreator<PoolOp<cl_data_t>>> __Pool_op(OpType_Pooling);
```

### OpenGL
#### 1. Shader
- 在[OpenGL/glsl](../source/backend/opengl/glsl)添加具体的shader(\*.glsl)，不用加文件头。
- 在[OpenGL目录](../source/backend/opengl/) 下执行 makeShader.py 

#### 2. Executor
在[OpenGL/execution/](../source/backend/opengl/execution/) 添加执行器，可参考 GLPool.h 和 GLPool.cpp

#### 3. 注册
OpenGL 不是用的抽象工厂方案，需要修改 [OpenGL](../source/backend/opengl/) 下的 GLBackend.cpp

如下：
```c++
switch (op->type()) {
    case OpType_Convolution:
        exe = new GLConvolution(op, this);
        break;
    case OpType_Pooling:
        exe = new GLPool(op->main_as_Pool(), this);
        break;
    case OpType_Eltwise:
        exe = new GLEltwise(op->main_as_Eltwise()->type(), inputs.size(), this);
        break;
    /*添加到后面即可*/

```

