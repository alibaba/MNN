[中文版本](AddOp_CN.md)

# Custom Op

Before adding Op, please refer to [Op Manual](OpList.md) to avoid unnecessary duplication.

In MNN, adding Op consists of the following steps:
1. [Add model description of Op](#Add model description of Op)
2. [Add model conversion of Op](#Add model conversion of Op)
3. [Calculate output tensor size of Op](#Calculate output tensor size of Op)
4. Add Op implementation (execution) corresponding to Backend([CPU](#CPU)|[Metal](#Metal)|[Vulkan](#Vulkan)|[OpenCL](#OpenCL)|[OpenGL](#OpenGL))

## Add model description of Op

First add the corresponding Op name and parameter description in [Model parameter description folder](../schema/default), as follows:
1. If the Op is from Caffe, add parameter description under [CaffeOp\.fbs](../schema/default/CaffeOp.fbs); if Op is from Tensorflow, then add under [TensorflowOp\.fbs](../schema Add under /default/TensorflowOp.fbs), for example:

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
2. Add the name of Op in the OpType list of [MNN\.fbs](../schema/default/MNN.fbs), and then add the parameter description name (the Pool after the table) in the OpParameter list.
> If Op has no parameters, you don't need step 1 above, just add the Op name to the OpType list.

## Add model conversion of Op

> Currently supports conversion of Tensorflow, Caffe, ONNX, and Tensorflow Lite model formats.

### Tensorflow Model Convert
1. Add xxxTf.cpp under [tensorflow directory](../tools/converter/source/tensorflow)
Implementation can refer to [PoolingTf\.cpp](../tools/converter/source/tensorflow/PoolingTf.cpp)
```c++
// Declare the specific Op conversion PoolingTf, inherit the Op conversion base class tfOpConverter
// Two methods
class PoolingTf : public tfOpConverter {                                             
    public:                                                                       
        virtual void run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph);
        PoolingTf() {}                                                                   
        virtual ~PoolingTf() {}                                                          
        virtual MNN::OpType opType();                                             
        virtual MNN::OpParameter type();                                          
}
// Macro definition
// DECLARE_OP_CONVERTER(PoolingTf);
```
> The run() function is used to parse the model's proto file to get the parameters, and then assign it to the flatbuffer custom parameter: you can call the function `find_attr_value(const tensorflow::NodeDef& node, const char* key, tensorflow::AttrValue& value)` to get the value of the corresponding parameters.

> virtual void run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph), The parameter srcNode holds the input/output node information, and the TmpNode can be found in the tempGraph according to the input and output nodes.
2. Add the corresponding tensorflow Op name to the MNN Op name mapping in OpMapper.hpp
```c++
{"MaxPool", MNN::OpType_Pooling},
{"AvgPool", MNN::OpType_Pooling},
```
3. Op with Const
- If Const is not treated as a parameter to this Op, but as a separate Op, you can ignore this step.
- If this Op wants to take its attached Const as a parameter, modify the function `_genMinGraph()` in the file `TmpGraph.cpp` and set the isCovered property of the corresponding Const node to true.

### Caffe Model Convert
1. Add xxx.cpp under [caffe directory](../tools/converter/source/caffe) 
Reference[Pool\.cpp](../tools/converter/source/caffe/Pool.cpp)
Parse the caffe parameter in the run function to get the parameters
> virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight). The parameter information of Op saveed in the parameters parameter, and the data parameters such as convolution/BN saved in the weight parameter.

### ONNX Model Convert
1. Add Op conversion xxxOnnx.cpp and complete the following three functions
```c++
MNN::OpType PoolingOnnx::opType() { return MNN::OpType_Pooling; }
MNN::OpParameter PoolingOnnx::type() { return MNN::OpParameter_Pool; }

void PoolingOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      std::vector<const onnx::TensorProto*> initializers)
```
> PoolingOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, std::vector<const onnx::TensorProto*> initializers), onnxNode is the onnx original node information, weight and other data information needs to be taken from initializers。

### Tensorflow Lite Model Convert
1. Add XXXTflite.cpp
```c++
DECLARE_OP_COVERTER(XXXTflite);

// Need to implement the following function
XXXTflite::opType(bool quantizedModel);
XXXTflite::type(bool quantizedModel);
XXXTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp, 
   const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
   const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
   const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet,
   bool quantizedModel)
// The interface has one more quantizedModel parameter than tensorflow. // If the quantizedModel is true, the model is a quantitative model and needs to be converted to quantified Op. 
// If it is false, it is converted to float Op.

// set input output index
dstOp->inputIndexes.resize(1);
dstOp->outputIndexes.resize(1);
dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
dstOp->outputIndexes[0] = tfliteOp->outputs[0];

// Register Op Conversion
using namespace tflite;
REGISTER_CONVERTER(SqueezeTflite, BuiltinOperator_SQUEEZE);
```

## Calculate output tensor size of Op

1. According to the dimension information of the input tensor, the dimension information of the output tensor is calculated, and the data type of the output tensor is set.
Inherit the base class SizeComputer, implement the `onComputeSize` function, return false if the input dimension information is unknown, and return true after the calculation is completed. For example Pooling：

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
        // Pool parameter info
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
        // input dimension information is unknown and return false
        if (outw <= 0 || outh <= 0) {
            return false;
        }
        // Pooling only change height and width
        output->buffer().dim[3].extent = outw;
        output->buffer().dim[2].extent = outh;

        return true;
    }
};

// Regist Shape Computer
REGISTER_SHAPE(XXXComputer, OpType_XXX);
```

## Add the corresponding Backend

### CPU
Add CPUXXX.hpp, CPUXXX.cpp under [CPU Directory](../Source/CPU) 。

1. Class Declaratio
Inherit the base class Execution and implement `onResize()` and `onExecute()`

```c++
class CPUPool : public Execution {
public:
    CPUPool(Backend *b, const Pool *parameter);
    virtual ~CPUPool() = default;

    // If you need to use cache when executing onExecute, apply in this function, otherwise you don't need to declare.
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    // Op execution function
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Pool *mParameter;

    Tensor mCacheLine;
};
```

2. Implement
Call `backend()->onAcquireBuffer(&mCacheLine, Backend::DYNAMIC)` in onResize() to request the cache and call `backend()->onReleaseBuffer(&mCacheLine, Backend::DYNAMIC)` to reclaim the cache for memory reuse.Do necessary input checks in onExecute() to find problem in advance, and return NO_ERROR correctly after the execution.

3. Implement CPU Op Creator, complete the registration

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

Add MetalXXX.hpp and MetalXXX.cpp under [Metal Directory](../source/backend/Metal) .

#### 1. Declare

Inherit the base class Execution, declare the constructor, the destruction, `onResize` and `onExecute` functions:

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

#### 2. Implement
* Unlike the CPU Tensor, which stores the data in the host pointer, the Metal data pointer is stored in the deviceId, and the deviceId stores the id<MTLBuffer>:

  ```c++
  auto buffer = (__bridge id<MTLBuffer>)(void *)tensor->deviceId();
  ```

* The parameters of Metal Op can be stored by id<MTLBuffer>. The buffer data type can be different from tensor. The buffer can even mix multiple data types, just ensure that the correct length is specified when creating. E.g:

  ```c++
  auto buffer = [context newDeviceBuffer:2 * sizeof(int) + 2 * sizeof(__fp16) access:CPUWriteOnly];
  ((__fp16 *)buffer.contents)[0] = mAlpha / mLocalSize;  // alpha
  ((__fp16 *)buffer.contents)[1] = mBeta;                // beta
  ((int *)buffer.contents)[1] = mLocalSize;              // local size
  ((int *)buffer.contents)[2] = inputs[0]->channel();    // channel
  ```

* When creating a buffer, you need to specify access control permissions. There are currently three permissions:
   CPUReadWrite, data is shared between CPU/GPU, generally used for device buffer;
   CPUWriteOnly, the data is not read after being written by the CPU, and is generally used for the parameter buffer;
   CPUTransparent, the data is only in the GPU, generally used in the heap buffer.

* **MNNMetalContext** has two sets of similar interfaces to create buffer, the difference is only in the life cycle of the data: the memory occupied by the device will not be reused in the single inference process; and the memory occupied by the heap is reused by other Ops after calling `-[MNNMetalContext releaseHeapBuffer:]`. In general, the heap will only be used with **CPUTransparent**.

  *heap only aviliable on iOS 10+，fall back to device on iOS 9*

* When using Metal, **It is forbidden to create device and library yourself if it is not a special case**. Loading the library and compiling the function are time-consuming behaviors, and **MNNMetalContext** does the necessary cache optimization. An example of executing Metal via context is as follows:

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

#### 3. Implement Metal Op Creator and complete registration

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
Add a specific shader(\*.comp) under [glsl directory](../source/backend/vulkan/execution/glsl), defaults of Pooling input memory layout is NC4HW4, so it is implemented by image, otherwise implement with buffer.

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

Then execute the `makeshader.py` script to compile the Shader.

#### 2. Class declaration
Add VulkanXXX.hpp and VulkanXXX.cpp under the directory [Operator](../source/backend/vulkan/execution/). Class declaration inherits VulkanBasicExecution:

```c++
class VulkanPool : public VulkanBasicExecution {
public:
    VulkanPool(const Op* op, Backend* bn);
    virtual ~VulkanPool();

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;
private:
    // GPU Shader needed parameters
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    // Pipeline
    const VulkanPipeline* mPoolPipeline;
    const Pool* mCommon;
    // Layout Descriptor Set
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
};
```

#### 3. Implement
To implement the function onEncode(), you first need to do a memory layout check (if NC4HW4, Shader uses image, otherwise use buffer), and return NO_ERROR after execution.

#### 4. Implement Vulkan Op Creator, complete registration
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
Add a specific kernel (\*.cl) in [cl directory](../source/backend/opencl/execution//cl), Pooling is implemented as image, and memory is sorted as `( H : batch * height, W : channel/4 * Width * channel4)`.

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

Add xxx.h and XXX.cpp under the directory [execution](../source/backend/opencl/execution/). The class declaration inherits the class Execution as follows

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
#### 3. Implement
Implement the function `onResize( ) (optional) `, `onExecute( )`, and return NO_ERROR after execution.

#### 4. Implement OpenCL Op Creator and register

as follows:

```c++
OpenCLCreatorRegister<TypedCreator<PoolOp<cl_data_t>>> __Pool_op(OpType_Pooling);
```

### OpenGL
#### 1. Shader
- Add a specific shader (\*.glsl) in [source/backend/OpenGL/glsl] (../source/backend/OpenGL/glsl) without adding a header.
- Execute makeShader.py under [source/backend/OpenGL/](../source/backend/OpenGL/)

#### 2. Executor
Add executors to [source/backend/OpenGL](../source/backend/OpenGL), refer to GLPool.h and GLPool.cpp

#### 3. Regist
OpenGL is not an abstract factory solution, you need to modify GLBackend.cpp under [source/backend/OpenGL](../source/backend/OpenGL)

As follows：
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
    /*Add to the back*/

```

