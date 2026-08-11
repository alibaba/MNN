#include "RangeExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void RangeKernel(T* output, T start, T delta, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < size) {
        output[index] = start + static_cast<T>(index) * delta;
    }
}

RangeExecution::RangeExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
}

ErrorCode RangeExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto output = outputs[0];
    
    mSize = 1;
    for (int i = 0; i < output->dimensions(); i++) {
        mSize *= output->length(i);
    }
    
    int threads = 256;
    int blocks = (mSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode RangeExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto output = outputs[0];
    auto op = mOp->main_as_Range();
    
    auto start = op->start();
    auto limit = op->limit();
    auto delta = op->delta();
    
    // Compute size from start, limit, delta
    mSize = static_cast<int>((limit - start) / delta);
    
    // Launch kernel based on data type
    if (op->type() == DataType_DT_FLOAT) {
        auto outputPtr = output->host<float>();
        RangeKernel<<<mDim3Grid, mDim3Block>>>(outputPtr, static_cast<float>(start), static_cast<float>(delta), mSize);
    } else if (op->type() == DataType_DT_INT32) {
        auto outputPtr = output->host<int>();
        RangeKernel<<<mDim3Grid, mDim3Block>>>(outputPtr, static_cast<int>(start), static_cast<int>(delta), mSize);
    } else {
        return COMPUTE_NO_SUPPORT;
    }
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class RangeCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new RangeExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<RangeCreator> gRangeRegistration(OpType_Range);

} // namespace MUSA
} // namespace MNN
