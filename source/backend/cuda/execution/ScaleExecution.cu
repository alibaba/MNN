#include "ScaleExecution.hpp"
#include "MNNCUDADefine.hpp"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void SCALE(const int total, const int channelsPack, const int dim, const T* in, T* out,
                        const float* scaleData, const float* biasData) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx = index / channelsPack;
        int c_idx = index % channelsPack;
        out[index] = (T)((float)in[index] * scaleData[c_idx] + biasData[c_idx]);
    }
}

ScaleExecution::ScaleExecution(const Scale* scale, Backend *backend) : Execution(backend) {
    int channel   = scale->scaleData()->size();
    mChannel = UP_DIV(channel, PACK_NUMBER);
    auto scaleBiasStorageSize = 2 * mChannel * PACK_NUMBER * sizeof(float);
    auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
    mScaleBiasStorage = staticPool->alloc(scaleBiasStorageSize);
    mDeviceScale = (uint8_t*)mScaleBiasStorage.first + mScaleBiasStorage.second;
    mDeviceBias = (uint8_t*)mDeviceScale + scaleBiasStorageSize / 2;
    cudaMemset(mDeviceScale, 0, scaleBiasStorageSize);
    {
        auto alphaData = scale->scaleData()->data();
        cudaMemcpy(mDeviceScale, alphaData, channel * sizeof(float), cudaMemcpyHostToDevice);
    }
    {
        auto biasData = scale->biasData()->data();
        if (nullptr != biasData) {
            cudaMemcpy(mDeviceBias, biasData, channel * sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}
ScaleExecution::~ScaleExecution() {
    auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    staticPool->free(mScaleBiasStorage);
}

ErrorCode ScaleExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    MNN_ASSERT(input->dimensions() >= 2);
    mArea      = input->length(0);
    for (int i = 2; i < input->dimensions(); ++i) {
        mArea *= input->length(i);
    }
    mCount = mChannel*mArea*PACK_NUMBER;
    //printf("mBatch:%d- mChannel:%d- mArea:%d- mCount:%d\n", mBatch,mChannel,mArea, mCount);
    return NO_ERROR;
}

ErrorCode ScaleExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        SCALE<<<block_num, threads_num>>>(mCount, mChannel*PACK_NUMBER, mArea, (const half *)input_addr, (half *)output_addr,
            (const float *)mDeviceScale, (const float *)mDeviceBias);
        return NO_ERROR;
    }
    SCALE<<<block_num, threads_num>>>(mCount, mChannel*PACK_NUMBER, mArea, (const float *)input_addr, (float *)output_addr,
        (const float *)mDeviceScale, (const float *)mDeviceBias);
    return NO_ERROR;
}

class ScaleCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_Scale();
        return new ScaleExecution(param, backend);
    }
};

static CUDACreatorRegister<ScaleCreator> __init(OpType_Scale);

}
}