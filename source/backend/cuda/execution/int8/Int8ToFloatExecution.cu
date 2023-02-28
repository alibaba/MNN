#include "Int8ToFloatExecution.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void INT8_2_FLOAT(const int total,
    const int channelsPackInt8,
    const int channelsPackFloat,
    const int channels, 
    const int8_t* in, 
    T* out,
    const float* scaleData, 
    const bool isSingle,
    const int8_t zeroPoint
) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx = index / channelsPackFloat;
        int c_idx = index % channelsPackFloat;
        if(c_idx >= channels) {
            out[index] = (T)0.0f;
            continue;
        }
        float scale = isSingle ? scaleData[0] : scaleData[c_idx];

        int idx_inp = nhw_idx * channelsPackInt8 + c_idx;
        out[index] = (T)((in[idx_inp] - zeroPoint) * scale);
    }
}

Int8ToFloatExecution::Int8ToFloatExecution(Backend *backend, const std::vector<Tensor *> &inputs, const MNN::Op *param) : Execution(backend) {
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    auto scale         = param->main_as_QuantizedFloatParam();

    if(scale == nullptr) {
        auto quantAttr = MNN::TensorUtils::getDescribe(inputs[0])->quantAttr;
        mZeroPoint = quantAttr->zero;

        mSingle = true;
        auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
        mScaleStorage = staticPool->alloc(PACK_NUMBER * sizeof(float));
        mScales = (void*)mScaleStorage.first + mScaleStorage.second;
        float tensorScale = quantAttr->scale;
        runtime->memcpy(mScales, &tensorScale, 1 * sizeof(float), MNNMemcpyHostToDevice);

    } else {
        const int scaleLen = scale->tensorScale()->size();
        mClipBits = scale->nbits();

        auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
        mScaleStorage = staticPool->alloc(UP_DIV(scaleLen, PACK_NUMBER) * PACK_NUMBER * sizeof(float));
        mScales = (void*)mScaleStorage.first + mScaleStorage.second;
        runtime->memset(mScales, 0, UP_DIV(scaleLen, PACK_NUMBER) * PACK_NUMBER * sizeof(float));

        if (1 == scaleLen) {
            mSingle = true;
            runtime->memcpy(mScales, scale->tensorScale()->data(), 1 * sizeof(float), MNNMemcpyHostToDevice);
        } else {
            runtime->memcpy(mScales, scale->tensorScale()->data(), scaleLen * sizeof(float), MNNMemcpyHostToDevice);
        }

        mZeroPoint = scale->zeroPoint();
    }
}
Int8ToFloatExecution::~Int8ToFloatExecution() {
    auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    staticPool->free(mScaleStorage);
}

ErrorCode Int8ToFloatExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    mArea      = input->length(0);
    mChannel = input->channel();
    for (int i = 2; i < input->dimensions(); ++i) {
        mArea *= input->length(i);
    }
    mCount = mArea * UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER;
    return NO_ERROR;
}

ErrorCode Int8ToFloatExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();

    auto channelPackInt8 = UP_DIV(mChannel, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    auto channelPackFloat = UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER;

    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        INT8_2_FLOAT<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const int8_t *)input_addr, (half *)output_addr,\
            (const float *)mScales, mSingle, mZeroPoint);
        checkKernelErrors;
    } else {
        INT8_2_FLOAT<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const int8_t *)input_addr, (float *)output_addr,\
            (const float *)mScales, mSingle, mZeroPoint);
        checkKernelErrors;
    }

    return NO_ERROR;
}

class Int8ToFloatCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new Int8ToFloatExecution(backend, inputs, op);
    }
};

static CUDACreatorRegister<Int8ToFloatCreator> __init(OpType_Int8ToFloat);

}
}