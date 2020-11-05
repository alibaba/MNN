#include "ScaleExecution.hpp"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void SCALE(const int n, const int channels, const int dim, const T* in, T* out,
                        const T* scaleData, const T* biasData) {
    CUDA_KERNEL_LOOP(index, n) {
        int c      = (index / dim) % channels;
        out[index] = in[index] * scaleData[c] + biasData[c];
    }
}

ScaleExecution::ScaleExecution(const Scale* scale, Backend *backend) : Execution(backend) {
    mChannel   = scale->scaleData()->size();

    scaleTensor.reset(Tensor::createDevice<float>({mChannel}));
    backend->onAcquireBuffer(scaleTensor.get(), Backend::STATIC);
    mDeviceScale = (void *)scaleTensor.get()->buffer().device;

    biasTensor.reset(Tensor::createDevice<float>({mChannel}));
    backend->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
    mDeviceBias = (void *)biasTensor.get()->buffer().device;
    
    MNN_ASSERT(nullptr != mDeviceScale);
    MNN_ASSERT(nullptr != mDeviceBias);
    {
        auto alphaData = scale->scaleData()->data();
        cudaMemcpy(mDeviceScale, alphaData, mChannel * sizeof(float), cudaMemcpyHostToDevice);
    }
    {
        auto biasData = scale->biasData()->data();
        if (nullptr != biasData) {
            MNN_ASSERT(mChannel == scale->biasData()->size());
            cudaMemcpy(mDeviceBias, biasData, mChannel * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            cudaMemset(mDeviceBias, 0, mChannel * sizeof(float));
        }
    }
}
ScaleExecution::~ScaleExecution() {
    if (nullptr != scaleTensor) {
        backend()->onReleaseBuffer(scaleTensor.get(), Backend::STATIC);
    }
    if (nullptr != biasTensor) {
        backend()->onReleaseBuffer(biasTensor.get(), Backend::STATIC);
    }
}

ErrorCode ScaleExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    mBatch     = input->length(0);
    MNN_ASSERT(mChannel == input->length(1));
    MNN_ASSERT(input->dimensions() >= 2);
    mArea      = 1;
    for (int i = 2; i < input->dimensions(); ++i) {
        mArea *= input->length(i);
    }
    mCount = mBatch*mChannel*mArea;
    //printf("mBatch:%d- mChannel:%d- mArea:%d- mCount:%d\n", mBatch,mChannel,mArea, mCount);
    return NO_ERROR;
}

ErrorCode ScaleExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();

    SCALE<<<block_num, threads_num>>>(mCount, mChannel, mArea, (const float *)input_addr, (float *)output_addr,
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