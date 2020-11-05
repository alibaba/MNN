#include "PReLUExecution.hpp"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void PRELU(const int n, const int channels, const int dim, const T* in, T* out,
                        const T* slopeData, int div_factor) {
    CUDA_KERNEL_LOOP(index, n) {
        int c      = (index / dim) % channels / div_factor;
        out[index] = in[index] > 0 ? in[index] : in[index]*slopeData[c];
    }
}

PReLUExecution::PReLUExecution(const PRelu* prelu, Backend *backend) : Execution(backend) {
    int slopCount = prelu->slope()->size();
    auto alphaData = prelu->slope()->data();
    preluTensor.reset(Tensor::createDevice<float>({slopCount}));
    backend->onAcquireBuffer(preluTensor.get(), Backend::STATIC);
    mDeviceSlope = (void *)preluTensor.get()->buffer().device;

    MNN_ASSERT(nullptr != mDeviceSlope);
    cudaMemcpy(mDeviceSlope, alphaData, slopCount * sizeof(float), cudaMemcpyHostToDevice);
    mIsChannelShared = slopCount == 1;

}
PReLUExecution::~PReLUExecution() {
    if (nullptr != preluTensor) {
        backend()->onReleaseBuffer(preluTensor.get(), Backend::STATIC);
    }
}

ErrorCode PReLUExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    mBatch     = input->length(0);
    mChannel   = input->length(1);
    MNN_ASSERT(input->dimensions() >= 2);
    mArea      = 1;
    for (int i = 2; i < input->dimensions(); ++i) {
        mArea *= input->length(i);
    }
    mCount = mBatch*mChannel*mArea;
    //printf("mBatch:%d- mChannel:%d- mArea:%d- mCount:%d\n", mBatch,mChannel,mArea, mCount);
    return NO_ERROR;
}

ErrorCode PReLUExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();
    int div_factor = mIsChannelShared ? mChannel : 1;
    PRELU<<<block_num, threads_num>>>(mCount, mChannel, mArea, (const float *)input_addr, (float *)output_addr,
        (const float *)mDeviceSlope, div_factor);
    return NO_ERROR;
}

class PReLUCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_PRelu();
        return new PReLUExecution(param, backend);
    }
};

static CUDACreatorRegister<PReLUCreator> __init(OpType_PReLU);

}
}