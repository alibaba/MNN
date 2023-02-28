#include "FloatToInt8Execution.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void FLOAT2INT8(const int total,
    const int channelsPackInt8,
    const int channelsPackFloat, 
    const int channels, 
    const T* in, 
    int8_t* out,
    const float* scaleData, 
    const bool isSingle, 
    const int8_t zeroPoint, 
    const int8_t clampMax, 
    const int8_t clampMin
) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx = index / channelsPackInt8;
        int c_idx = index % channelsPackInt8;
        if(c_idx >= channels) {
            out[index] = 0;
            continue;
        }
        float scale = isSingle ? scaleData[0] : scaleData[c_idx];
        int idx_inp = nhw_idx * channelsPackFloat + c_idx;
        int res = __float2int_rn((float)(in[idx_inp]) * scale) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[index] = res;
    }
}

FloatToInt8Execution::FloatToInt8Execution(Backend *backend, const std::vector<Tensor *> &inputs, const MNN::Op *param) : Execution(backend) {
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    auto scale         = param->main_as_QuantizedFloatParam();

    if(scale == nullptr) {
        auto quantAttr = MNN::TensorUtils::getDescribe(inputs[0])->quantAttr;
        mZeroPoint = quantAttr->zero;
        mClampMax  = quantAttr->max;
        mClampMin  = quantAttr->min;

        mSingle = true;
        auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
        mScaleStorage = staticPool->alloc(INT8_PACK_NUMBER * sizeof(float));
        mScales = (void*)mScaleStorage.first + mScaleStorage.second;

        float tensorScale = quantAttr->scale;
        runtime->memcpy(mScales, &tensorScale, 1 * sizeof(float), MNNMemcpyHostToDevice);
    } else {
        const int scaleLen = scale->tensorScale()->size();
        mClipBits = scale->nbits();

        auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
        mScaleStorage = staticPool->alloc(UP_DIV(scaleLen, INT8_PACK_NUMBER) * INT8_PACK_NUMBER * sizeof(float));
        mScales = (void*)mScaleStorage.first + mScaleStorage.second;
        runtime->memset(mScales, 0, UP_DIV(scaleLen, INT8_PACK_NUMBER) * INT8_PACK_NUMBER * sizeof(float));


        if (1 == scaleLen) {
            mSingle = true;
            runtime->memcpy(mScales, scale->tensorScale()->data(), 1 * sizeof(float), MNNMemcpyHostToDevice);
        } else {
            runtime->memcpy(mScales, scale->tensorScale()->data(), scaleLen * sizeof(float), MNNMemcpyHostToDevice);
        }

        mZeroPoint = scale->zeroPoint();
        mClampMin = scale->clampMin();
        mClampMax = scale->clampMax();
    }
}
FloatToInt8Execution::~FloatToInt8Execution() {
    auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    staticPool->free(mScaleStorage);
}

ErrorCode FloatToInt8Execution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    mArea      = input->length(0);
    mChannel = input->channel();
    for (int i = 2; i < input->dimensions(); ++i) {
        mArea *= input->length(i);
    }
    mCount = mArea * UP_DIV(mChannel, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    //printf("mBatch:%d- mChannel:%d- mArea:%d- mCount:%d\n", mBatch,mChannel,mArea, mCount);
    return NO_ERROR;
}

ErrorCode FloatToInt8Execution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();

    auto channelPackInt8 = UP_DIV(mChannel, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    auto channelPackFloat = UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER;
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        FLOAT2INT8<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const half *)input_addr, (int8_t *)output_addr,\
            (const float *)mScales, mSingle, mZeroPoint, mClampMax, mClampMin);
        checkKernelErrors;
    } else {
        FLOAT2INT8<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const float *)input_addr, (int8_t *)output_addr,\
            (const float *)mScales, mSingle, mZeroPoint, mClampMax, mClampMin);
        checkKernelErrors;
    }

    return NO_ERROR;
}

class FloatToInt8Creator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new FloatToInt8Execution(backend, inputs, op);
    }
};

static CUDACreatorRegister<FloatToInt8Creator> __init(OpType_FloatToInt8);

}
}