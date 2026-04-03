#include "CastExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename InputT, typename OutputT>
__global__ void CastKernel(const InputT* input, OutputT* output, int totalSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < totalSize) {
        output[index] = static_cast<OutputT>(input[index]);
    }
}

CastExecution::CastExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_Cast();
}

ErrorCode CastExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    mTotalSize = 1;
    for (int i = 0; i < input->dimensions(); i++) {
        mTotalSize *= input->length(i);
    }
    
    int threads = 256;
    int blocks = (mTotalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode CastExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    auto srcType = mOp->srcT();
    auto dstType = mOp->dstT();
    
    // Handle common type conversions
    if (srcType == DataType_DT_FLOAT && dstType == DataType_DT_INT32) {
        auto inputPtr = input->host<float>();
        auto outputPtr = output->host<int>();
        CastKernel<<<mDim3Grid, mDim3Block>>>(inputPtr, outputPtr, mTotalSize);
    } else if (srcType == DataType_DT_FLOAT && dstType == DataType_DT_INT8) {
        auto inputPtr = input->host<float>();
        auto outputPtr = output->host<int8_t>();
        CastKernel<<<mDim3Grid, mDim3Block>>>(inputPtr, outputPtr, mTotalSize);
    } else if (srcType == DataType_DT_INT32 && dstType == DataType_DT_FLOAT) {
        auto inputPtr = input->host<int>();
        auto outputPtr = output->host<float>();
        CastKernel<<<mDim3Grid, mDim3Block>>>(inputPtr, outputPtr, mTotalSize);
    } else if (srcType == DataType_DT_INT8 && dstType == DataType_DT_FLOAT) {
        auto inputPtr = input->host<int8_t>();
        auto outputPtr = output->host<float>();
        CastKernel<<<mDim3Grid, mDim3Block>>>(inputPtr, outputPtr, mTotalSize);
    } else if (srcType == DataType_DT_FLOAT && dstType == DataType_DT_FLOAT) {
        auto inputPtr = input->host<float>();
        auto outputPtr = output->host<float>();
        CastKernel<<<mDim3Grid, mDim3Block>>>(inputPtr, outputPtr, mTotalSize);
    } else {
        // For unsupported types, return error
        return COMPUTE_NO_SUPPORT;
    }
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class CastCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new CastExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<CastCreator> gCastRegistration(OpType_Cast);

} // namespace MUSA
} // namespace MNN
