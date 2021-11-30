#include "ScatterNdExecution.hpp"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename T>
__global__ void SETZERO(const int n, T* outputPtr) {
    CUDA_KERNEL_LOOP(index, n) {
        outputPtr[index] = (T)0;
    }
}

template<typename T>
__global__ void SCATTERND(const int n, const int indicesLastDim, const int accNumber, const int* indicesPtr,
    const T* updatesPtr, T* outputPtr, const int32_t* dimsToCount) {
    CUDA_KERNEL_LOOP(index, n) {
        int pos = 0;
        for (int j = 0; j < indicesLastDim; ++j) {
            auto curIndex = (int)indicesPtr[index * indicesLastDim + j];
            // MNN_ASSERT(curIndex >= 0 && curIndex < output->length(j));
            pos += curIndex * dimsToCount[j];
        }
        for (int k = 0; k < accNumber; ++k) {
            float updateValue = updatesPtr[index * accNumber + k];
            atomicAdd(outputPtr + pos + k, updateValue);
        }
    }
}

ScatterNdExecution::ScatterNdExecution(Backend *backend) : Execution(backend) {
    // Do nothing
}
ScatterNdExecution::~ScatterNdExecution() {
    // Do nothing
}

ErrorCode ScatterNdExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 3);
    MNN_ASSERT(outputs.size() == 1);

    auto indices               = inputs[0];
    auto updates               = inputs[1];
    auto shape                 = inputs[2];
    auto output                = outputs[0];
    const int indicesDimension = indices->dimensions();
    mIndicesLastDim    = indices->length(indicesDimension - 1);
    mIndexes           = indices->elementSize() / mIndicesLastDim;

    mAccNumber = 1;
    for (int i = indicesDimension - 1; i < updates->dimensions(); ++i) {
        mAccNumber *= updates->length(i);
    }

    const int outputElementSize = output->elementSize();
    mOutElementSize     = outputElementSize;
    int remainSize              = outputElementSize;
    std::vector<int> temp(mIndicesLastDim, 0);
    for (int i = 0; i < mIndicesLastDim; ++i) {
        temp[i]    = remainSize / output->length(i);
        remainSize = temp[i];
    }

    //save dimToCount to Device
    dimsTensor.reset(Tensor::createDevice<int>({mIndicesLastDim}));
    backend()->onAcquireBuffer(dimsTensor.get(), Backend::STATIC);
    mDimsToCount = (void *)dimsTensor.get()->buffer().device;
    cuda_check(cudaMemcpy(mDimsToCount, temp.data(), mIndicesLastDim*sizeof(int), cudaMemcpyHostToDevice));

    return NO_ERROR;
}

ErrorCode ScatterNdExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num0 = runtime->blocks_num(mOutElementSize);
    int block_num1 = runtime->blocks_num(mIndexes);
    int threads_num = runtime->threads_num();
    auto input_addr0 = (void*)inputs[0]->deviceId();
    auto input_addr1 = (void*)inputs[1]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();

    //printf("mOutElementSize:%d- mIndexes:%d- mIndicesLastDim:%d- mAccNumber:%d\n", mOutElementSize,mIndexes,mIndicesLastDim, mAccNumber);

    SETZERO<<<block_num0, threads_num>>>(mOutElementSize, (float*)output_addr);
    SCATTERND<<<block_num1, threads_num>>>(mIndexes, mIndicesLastDim, mAccNumber,
        (const int*)input_addr0, (const float*)input_addr1, (float*)output_addr, (const int32_t*)mDimsToCount);
    return NO_ERROR;
}

class ScatterNdCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if(inputs.size() != 3) {
            MNN_PRINT("CUDA ScatterNd inputs size:%d not support, back to CPU\n", inputs.size());
            return nullptr;
        }
        return new ScatterNdExecution(backend);
    }
};

static CUDACreatorRegister<ScatterNdCreator> __init(OpType_ScatterNd);

}
}
