#include "ArgMaxExecution.hpp"
#include "ArgMinExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

#define ARG_REDUCE_NUM 256

template <typename T>
__global__ void ARGMAX_FIRST_STEP(const int count, const int outside, const int inside,
                            const int totalDims, const int dims, const int numDims,
                            const T *input, T *outputData, int *outputIndex
    ) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x) {
        const int idx_in = index % inside;
        const int tmp = index / inside;
        const int idx_out = tmp % outside;
        const int idx_num_dim = tmp / outside;

        const int idx_output = (idx_out * numDims + idx_num_dim) * inside + idx_in;
        const T* inpPtr = input + (idx_out * totalDims + idx_num_dim * dims) * inside + idx_in;
        int maxIndex = idx_num_dim * dims;
        T maxValue = inpPtr[0 * inside];
        for(int j=1; j<dims; j++) {
            const int idx_access = idx_num_dim * dims + j;
            if(idx_access < totalDims) {
                T value = inpPtr[j * inside];
                if(maxValue < value) {
                    maxIndex = idx_access;
                    maxValue = value;
                }
            }
        }
        outputData[idx_output] = maxValue;
        outputIndex[idx_output] = maxIndex;
    }

}

template <typename T>
__global__ void ARGMAX_SECOND_STEP(const int count, const int outside, const int inside, const int dims,
                            const T *inputData, const int *inputIndex, int *outputIndex
    ) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x) {
        const int idx_in = index % inside;
        const int idx_out = index / inside;

        int idx_output = idx_out * inside + idx_in;
        const T* inpPtr = inputData + idx_out * dims * inside + idx_in;
        const int* baseInputIndex = inputIndex + idx_out * dims * inside + idx_in;
        int maxIndex = baseInputIndex[0];
        T maxValue = inpPtr[0 * inside];
        for(int j=1; j<dims; j++) {
            T value = inpPtr[j * inside];
            if(maxValue < value) {
                maxIndex = baseInputIndex[j];
                maxValue = value;
            }
        }
        outputIndex[idx_output] = maxIndex;
    }
}

template <typename T>
__global__ void ARGMAX(const int count, const int outside, const int inside, const int dim,
                         const T *input, int *output) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        const int idx_out = i / inside;
        const int idx_in = i % inside;

        int* outPtr = output + idx_out * inside + idx_in;
        const T* inpPtr = input + idx_out * inside * dim + idx_in;
        int index = 0;
        T maxValue = inpPtr[0 * inside];
        for(int j=1; j<dim; j++) {
            T value = inpPtr[j * inside];
            if(maxValue < value) {
                index = j;
                maxValue = value;
            }
        }
        outPtr[0] = index;
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
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
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

    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);
    mSplitKernel = (mDim > ARG_REDUCE_NUM);
    if(mSplitKernel) {
        mSecondArgLen = (mDim + ARG_REDUCE_NUM - 1) / ARG_REDUCE_NUM;
        auto buffer_data = pool->alloc(mOutside * mInside * mSecondArgLen * bytes);
        mTempDataBuffer = (void*)(buffer_data.ptr());
        auto buffer_index = pool->alloc(mOutside * mInside * mSecondArgLen * sizeof(int32_t));
        mTempIndexBuffer = (void*)(buffer_index.ptr());
        pool->free(buffer_data);
        pool->free(buffer_index);
    }
    return NO_ERROR;
}

ErrorCode ArgMaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend *>(backend())->getCUDARuntime();

    auto input = (void *)inputs[0]->deviceId();
    auto output = (void *)outputs[0]->deviceId();
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);

    if(mSplitKernel) {
        if(bytes == 4) {
            // First Step
            {
                int count = mOutside * mInside * mSecondArgLen;
                int block_num = runtime->blocks_num(count);
                int thread_num = runtime->threads_num();
                ARGMAX_FIRST_STEP<<<block_num, thread_num>>>(count, mOutside, mInside, mDim, ARG_REDUCE_NUM, mSecondArgLen, \
                    (const float*)input, (float *)mTempDataBuffer, (int *)mTempIndexBuffer);
                checkKernelErrors;
            }
            // Second Step
            {
                int count = mOutside * mInside;
                int block_num = runtime->blocks_num(count);
                int thread_num = runtime->threads_num();
                ARGMAX_SECOND_STEP<<<block_num, thread_num>>>(count, mOutside, mInside, mSecondArgLen, \
                    (const float*)mTempDataBuffer, (const int *)mTempIndexBuffer, (int *)output);
                checkKernelErrors;
            }
        } else {
            // First Step
            {
                int count = mOutside * mInside * mSecondArgLen;
                int block_num = runtime->blocks_num(count);
                int thread_num = runtime->threads_num();
                ARGMAX_FIRST_STEP<<<block_num, thread_num>>>(count, mOutside, mInside, mDim, ARG_REDUCE_NUM, mSecondArgLen, \
                    (const half*)input, (half *)mTempDataBuffer, (int *)mTempIndexBuffer);
                checkKernelErrors;
            }
            // Second Step
            {
                int count = mOutside * mInside;
                int block_num = runtime->blocks_num(count);
                int thread_num = runtime->threads_num();
                ARGMAX_SECOND_STEP<<<block_num, thread_num>>>(count, mOutside, mInside, mSecondArgLen, \
                    (const half*)mTempDataBuffer, (const int *)mTempIndexBuffer, (int *)output);
                checkKernelErrors;
            }
        }

    } else {
        int count = mOutside * mInside;
        int block_num = runtime->blocks_num(count);
        int thread_num = runtime->threads_num();

        if(bytes == 4) {
            ARGMAX<<<block_num, thread_num>>>(count, mOutside, mInside, mDim, (const float*)input,(int *)output);
            checkKernelErrors;
        } else {
            ARGMAX<<<block_num, thread_num>>>(count, mOutside, mInside, mDim, (const half*)input,(int *)output);
            checkKernelErrors;
        }
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
