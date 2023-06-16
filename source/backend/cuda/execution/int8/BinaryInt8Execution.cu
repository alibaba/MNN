
//
//  BinaryInt8Execution.cu
//  MNN
//
//  Created by MNN on 2023/05/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT
#include "BinaryInt8Execution.hpp"

namespace MNN {
namespace CUDA {

#define BINARY_INT8_FUNC(Name, Func)\
__global__ void BINARY_INT8_##Name(\
    const int maxCount,\
    const int8_t* input0_addr,\
    const float input0_scale,\
    const int8_t* input1_addr,\
    const float  input1_scale,\
    int8_t* output_addr,\
    const float output_scale,\
    const int s0,\
    const int s1\
) {\
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {\
        float x = (float)input0_addr[index*s0] * input0_scale;\
        float y = (float)input1_addr[index*s1] * input1_scale;\
        float val = Func;\
        int res = __float2int_rn(output_scale * val);\
        res = min(res, 127);\
        res = max(res, -128);\
        output_addr[index] = res;\
    }\
}\

#define BINARY_INT8_CHANNEL_FUNC(Name, Func)\
__global__ void BINARY_INT8_CHANNELWISE_##Name(\
    const int maxCount,\
    const int channelPack,\
    const int8_t* input0_addr,\
    const float* input0_scale,\
    const int8_t* input1_addr,\
    const float* input1_scale,\
    int8_t* output_addr,\
    const float* output_scale,\
    DivModFast d_cp\
) {\
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {\
        int cpIndex, nhwIndex;\
        d_cp.divmod(index, nhwIndex, cpIndex);\
        float x = (float)input0_addr[index] * input0_scale[cpIndex];\
        float y = (float)input1_addr[index] * input1_scale[cpIndex];\
        float val = Func;\
        int res = __float2int_rn(output_scale[cpIndex] * val);\
        res = min(res, 127);\
        res = max(res, -128);\
        output_addr[index] = res;\
    }\
}\

#define sign(y) ((y) > 0 ? 1 : ((y) < 0 ? -1 : 0))

BINARY_INT8_FUNC(ADD, x+y);
BINARY_INT8_FUNC(SUB, x-y);
BINARY_INT8_FUNC(MUL, x*y);
BINARY_INT8_FUNC(DIV, x/y);
BINARY_INT8_FUNC(REALDIV, (float)sign(y) * x / max(abs(y), 0.0000001));
BINARY_INT8_FUNC(MINIMUM, min(x, y));
BINARY_INT8_FUNC(MAXIMUM, max(x, y));
BINARY_INT8_FUNC(GREATER, x > y ? 1 : 0);
BINARY_INT8_FUNC(LESS, x < y ? 1 : 0);
BINARY_INT8_FUNC(LESS_EQUAL, x <= y ? 1 : 0);
BINARY_INT8_FUNC(GREATER_EQUAL, x >= y ? 1 : 0);
BINARY_INT8_FUNC(EQUAL, x == y ? 1 : 0);
BINARY_INT8_FUNC(NOTEQUAL, x != y ? 1 : 0);
BINARY_INT8_FUNC(FLOORDIV, floor(x / y));
BINARY_INT8_FUNC(FLOORMOD, x - floor(x / y) * y);
BINARY_INT8_FUNC(SquaredDifference, (x-y)*(x-y));
BINARY_INT8_FUNC(POW, pow(x, y));
BINARY_INT8_FUNC(ATAN2, atan2(x, y));
BINARY_INT8_FUNC(LOGICALOR, (x || y) ? 1 : 0);

BINARY_INT8_CHANNEL_FUNC(ADD, x+y);
BINARY_INT8_CHANNEL_FUNC(SUB, x-y);
BINARY_INT8_CHANNEL_FUNC(MUL, x*y);
BINARY_INT8_CHANNEL_FUNC(DIV, x/y);
BINARY_INT8_CHANNEL_FUNC(REALDIV, (float)sign(y) * x / max(abs(y), 0.0000001));
BINARY_INT8_CHANNEL_FUNC(MINIMUM, min(x, y));
BINARY_INT8_CHANNEL_FUNC(MAXIMUM, max(x, y));
BINARY_INT8_CHANNEL_FUNC(GREATER, x > y ? 1 : 0);
BINARY_INT8_CHANNEL_FUNC(LESS, x < y ? 1 : 0);
BINARY_INT8_CHANNEL_FUNC(LESS_EQUAL, x <= y ? 1 : 0);
BINARY_INT8_CHANNEL_FUNC(GREATER_EQUAL, x >= y ? 1 : 0);
BINARY_INT8_CHANNEL_FUNC(EQUAL, x == y ? 1 : 0);
BINARY_INT8_CHANNEL_FUNC(NOTEQUAL, x != y ? 1 : 0);
BINARY_INT8_CHANNEL_FUNC(FLOORDIV, floor(x / y));
BINARY_INT8_CHANNEL_FUNC(FLOORMOD, x - floor(x / y) * y);
BINARY_INT8_CHANNEL_FUNC(SquaredDifference, (x-y)*(x-y));
BINARY_INT8_CHANNEL_FUNC(POW, pow(x, y));
BINARY_INT8_CHANNEL_FUNC(ATAN2, atan2(x, y));
BINARY_INT8_CHANNEL_FUNC(LOGICALOR, (x || y) ? 1 : 0);

BinaryInt8Execution::BinaryInt8Execution(const MNN::Op* op, Backend *backend, int activationType) : Execution(backend) {
    mIsEltwiseInt8 = op->type() == OpType_EltwiseInt8;
    if (!mIsEltwiseInt8) {
        mType = op->main_as_BinaryOp()->opType();
        return;
    }

    auto eltwise = op->main_as_Eltwise();
    switch (eltwise->type()) {
        case EltwiseType_PROD:
            mType = BinaryOpOperation_MUL;
            break;
        case EltwiseType_SUM:
            mType = BinaryOpOperation_ADD;
            break;
        case EltwiseType_MAXIMUM:
            mType = BinaryOpOperation_MAXIMUM;
            break;
        default:
            MNN_PRINT("Unsupported eltwise type %d!\n", eltwise->type());
            break;
    }

    mActivationType = activationType;

    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    auto param    = op->main_as_EltwiseInt8();

    auto copyData = [=](std::shared_ptr<Tensor>& tensor, const QuantizedFloatParam* scale) {
        const int size = scale->tensorScale()->size();
        const int size_pack = UP_DIV(size, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
        tensor.reset(Tensor::createDevice<float>({size_pack}));
        bool success =  static_cast<CUDABackend*>(backend)->onAcquireBuffer(tensor.get(), Backend::STATIC);
        if (!success) {
            return;
        }
        runtime->memset((void *)tensor.get()->buffer().device, 0, size_pack * sizeof(float));
        runtime->memcpy((void *)tensor.get()->buffer().device, scale->tensorScale()->data(), size * sizeof(float), MNNMemcpyHostToDevice);
    };

    copyData(mInput0ScalesTensor, param->inputQuan0());
    copyData(mInput1ScalesTensor, param->inputQuan1());
    copyData(mOutputScalesTensor, param->outputQuan());
}
BinaryInt8Execution::~BinaryInt8Execution(){
    // Do nothing
}
ErrorCode BinaryInt8Execution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    // MNN_PRINT("isEltwiseInt8:%d scale inp0 inp1, out :%f %f %f, format:%d\n", mIsEltwiseInt8, MNN::TensorUtils::getDescribe(inputs[0])->quantAttr->scale, MNN::TensorUtils::getDescribe(inputs[1])->quantAttr->scale, MNN::TensorUtils::getDescribe(outputs[0])->quantAttr->scale, MNN::TensorUtils::getDescribe(inputs[0])->dimensionFormat);
    auto count = CUDABackend::realSize(outputs[0]);
    auto inputS0 = CUDABackend::realSize(inputs[0]);
    auto inputS1 = CUDABackend::realSize(inputs[1]);
    int s0 = inputS0 == 1 ? 0 : 1;
    int s1 = inputS1 == 1 ? 0 : 1;

    // MNN_PRINT("BinaryInt8: inp0:%d inp1:%d out:%d\n", inputS0, inputS1, count);
    auto input0_addr  = inputs[0]->deviceId();
    auto input1_addr  = inputs[1]->deviceId();
    auto output_addr  = outputs[0]->deviceId();

    const int channel = outputs[0]->channel();
    const int channel_pack = UP_DIV(channel, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    DivModFast cpD(channel_pack);

    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();

    #define COMPUTE(TYPE)\
    if (mType == MNN::BinaryOpOperation_##TYPE ) {\
        BINARY_INT8_##TYPE<<<block_num, threads_num>>>(count,\
            (const int8_t*)input0_addr, TensorUtils::getDescribe(inputs[0])->quantAttr->scale,\
            (const int8_t*)input1_addr, TensorUtils::getDescribe(inputs[1])->quantAttr->scale,\
            (int8_t*)output_addr, 1.0 / TensorUtils::getDescribe(outputs[0])->quantAttr->scale,\
            s0, s1);\
        checkKernelErrors;\
    }\

    if(!mIsEltwiseInt8) {
        COMPUTE(ADD);
        COMPUTE(SUB);
        COMPUTE(MUL);
        COMPUTE(DIV);
        COMPUTE(REALDIV);
        COMPUTE(MINIMUM);
        COMPUTE(MAXIMUM);
        COMPUTE(GREATER);
        COMPUTE(LESS);
        COMPUTE(LESS_EQUAL);
        COMPUTE(GREATER_EQUAL);
        COMPUTE(EQUAL);
        COMPUTE(NOTEQUAL);
        COMPUTE(FLOORDIV);
        COMPUTE(FLOORMOD);
        COMPUTE(POW);
        COMPUTE(SquaredDifference);
        COMPUTE(ATAN2);
        COMPUTE(LOGICALOR);
    } else {
        auto input0_scale = mInput0ScalesTensor.get()->buffer().device; 
        auto input1_scale = mInput1ScalesTensor.get()->buffer().device; 
        auto output_scale = mOutputScalesTensor.get()->buffer().device; 

        #define COMPUTE_CHANNELWISE(TYPE)\
        if (mType == MNN::BinaryOpOperation_##TYPE ) {\
            BINARY_INT8_CHANNELWISE_##TYPE<<<block_num, threads_num>>>(count, channel_pack,\
                (const int8_t*)input0_addr, (const float*)input0_scale,\
                (const int8_t*)input1_addr, (const float*)input1_scale,\
                (int8_t*)output_addr, (const float*)output_scale, cpD);\
            checkKernelErrors;\
            return NO_ERROR;\
        }\

        COMPUTE_CHANNELWISE(ADD);
        COMPUTE_CHANNELWISE(SUB);
        COMPUTE_CHANNELWISE(MUL);
        COMPUTE_CHANNELWISE(DIV);
        COMPUTE_CHANNELWISE(REALDIV);
        COMPUTE_CHANNELWISE(MINIMUM);
        COMPUTE_CHANNELWISE(MAXIMUM);
        COMPUTE_CHANNELWISE(GREATER);
        COMPUTE_CHANNELWISE(LESS);
        COMPUTE_CHANNELWISE(LESS_EQUAL);
        COMPUTE_CHANNELWISE(GREATER_EQUAL);
        COMPUTE_CHANNELWISE(EQUAL);
        COMPUTE_CHANNELWISE(NOTEQUAL);
        COMPUTE_CHANNELWISE(FLOORDIV);
        COMPUTE_CHANNELWISE(FLOORMOD);
        COMPUTE_CHANNELWISE(POW);
        COMPUTE_CHANNELWISE(SquaredDifference);
        COMPUTE_CHANNELWISE(ATAN2);
        COMPUTE_CHANNELWISE(LOGICALOR);
    }

    return NO_ERROR;
}
class BinaryInt8Creator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        return new BinaryInt8Execution(op, backend);
    }
};

static CUDACreatorRegister<BinaryInt8Creator> __init(OpType_EltwiseInt8);
}
}
#endif