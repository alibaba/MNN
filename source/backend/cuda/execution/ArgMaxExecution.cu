#include "ArgMaxExecution.hpp"
#include "ArgMinExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

template <typename T>
__global__ void ARGMAX(const int count, const int outside, const int inside, const int dim,
                         const T *input, int *output) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        const int o = i / inside;
        const int n = i % inside;

        int* outPtr = output + inside * o;
        const T* inpPtr = input + inside * dim * o;
        int index = 0;
        T maxValue = inpPtr[n + 0 * inside];
        for(int j=1; j<dim; j++) {
            T value = inpPtr[n + j * inside];
            if(maxValue < value) {
                index = j;
                maxValue = value;
            }
        }
        outPtr[n] = index;
    }

    return;
}
ArgMaxExecution::ArgMaxExecution(const Op* op, Backend *backend) : Execution(backend) {
    mOp = op;
    mAxis = mOp->main_as_ArgMax()->axis();
}

ArgMaxExecution::~ArgMaxExecution(){
    // Do nothing
}

ErrorCode ArgMaxExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output  = outputs[0];

    if (mAxis < 0) {
        mAxis = input->dimensions() + mAxis;
    }

    mInside = 1;
    mOutside = 1;
    for (int i=0; i<mAxis; ++i) {
        mOutside *= input->length(i);
    }
    for (int i=mAxis+1; i<input->dimensions(); ++i) {
        mInside *= input->length(i);
    }
    mDim = input->length(mAxis);
    return NO_ERROR;
}

ErrorCode ArgMaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend *>(backend())->getCUDARuntime();

    auto input = (void *)inputs[0]->deviceId();
    auto output = (void *)outputs[0]->deviceId();
    int count = mOutside * mInside;
    int block_num = runtime->blocks_num(count);
    int thread_num = runtime->threads_num();

    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);

    if(bytes == 4) {
        ARGMAX<<<block_num, thread_num>>>(count, mOutside, mInside, mDim, (const float*)input,(int *)output);
        checkKernelErrors;
    } else {
        ARGMAX<<<block_num, thread_num>>>(count, mOutside, mInside, mDim, (const half*)input,(int *)output);
        checkKernelErrors;
    }

    return NO_ERROR;
}
class ArgMaxCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto input = inputs[0];
        if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        if (op->type() == OpType_ArgMax) {
            return new ArgMaxExecution(op, backend);
        } else {
            return new ArgMinExecution(op, backend);
        }

    }
};

static CUDACreatorRegister<ArgMaxCreator> __init(OpType_ArgMax);
static CUDACreatorRegister<ArgMaxCreator> __init_op2(OpType_ArgMin);

}
}
