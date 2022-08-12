#include "GatherV2Execution.hpp"
namespace MNN {
namespace CUDA {

template <typename T>
__global__ void GATHERV2(const int count, const int outside, const int inside, const int iNum, const int oNum,
                         const T *input, const int* indice, T *output) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int x = i % inside;
        int y = i / inside;
        const int o = y / oNum;
        const int n = y % oNum;

        T* outPtr = output + inside * oNum * o;
        const T* inpPtr = input + inside * iNum * o;
        outPtr[n*inside+x] = inpPtr[indice[n]*inside+x];
    }
    return;
}

GatherV2Execution::GatherV2Execution(const Op* op, Backend *backend) : Execution(backend) {
    mOp = op;
}

GatherV2Execution::~GatherV2Execution(){
    // Do nothing
}

ErrorCode GatherV2Execution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto params  = inputs[0];
    mAxis = 0;
    if (mOp->main_type() == OpParameter_Axis) {
        mAxis = mOp->main_as_Axis()->axis();
    }
    MNN_ASSERT(mAxis > -params->buffer().dimensions && mAxis < params->buffer().dimensions);

    if (mAxis < 0) {
        mAxis = params->buffer().dimensions + mAxis;
    }

    auto indices = inputs[1];
    auto output  = outputs[0];
    mOutNum         = indices->elementSize();
    mInside = 1;
    mOutside = 1;
    for (int i=0; i<mAxis; ++i) {
        mOutside *= params->length(i);
    }
    for (int i=mAxis+1; i<params->dimensions(); ++i) {
        mInside *= params->length(i);
    }
    mInpNum = params->length(mAxis);

    return NO_ERROR;
}

ErrorCode GatherV2Execution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend *>(backend())->getCUDARuntime();

    auto params = (void *)inputs[0]->deviceId();
    auto indices = (void *)inputs[1]->deviceId();
    auto output = (void *)outputs[0]->deviceId();

    if (inputs.size() == 3) {
        cudaMemcpy(&mAxis, (void *)inputs[2]->deviceId(), sizeof(int), cudaMemcpyDeviceToHost);

        auto input0  = inputs[0];
        MNN_ASSERT(mAxis > -input0->dimensions() && mAxis < input0->dimensions());
        if (mAxis < 0) {
            mAxis = input0->dimensions() + mAxis;
        }
    
        mInside = 1;
        mOutside = 1;
        for (int i=0; i<mAxis; ++i) {
            mOutside *= input0->length(i);
        }
        for (int i=mAxis+1; i<input0->dimensions(); ++i) {
            mInside *= input0->length(i);
        }
        mInpNum = input0->length(mAxis);
    }

    int count = mOutside * mOutNum * mInside;

    int block_num = runtime->blocks_num(count);
    int thread_num = runtime->threads_num();
    //printf("count:%d, mOutside:%d, mInside:%d, mInpNum:%d, mOutNum:%d\n", count, mOutside, mInside, mInpNum, mOutNum);
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);

    if(bytes == 4) {
        GATHERV2<<<block_num, thread_num>>>(count, mOutside, mInside, mInpNum, mOutNum, 
                (const float*)params, (int *)indices, (float *)output);
        checkKernelErrors;
    } else {
        GATHERV2<<<block_num, thread_num>>>(count, mOutside, mInside, mInpNum, mOutNum, 
                (const half*)params, (int *)indices, (half *)output);
        checkKernelErrors;
    }
    return NO_ERROR;
}
class GatherV2Creator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new GatherV2Execution(op, backend);
    }
};

static CUDACreatorRegister<GatherV2Creator> __init2(OpType_GatherV2);
static CUDACreatorRegister<GatherV2Creator> __init(OpType_Gather);

}
}