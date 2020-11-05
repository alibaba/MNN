#include "ReductionExecution.hpp"

namespace MNN {
namespace CUDA {

ReductionExecution::ReductionExecution(ReductionType opType, int axis, Backend *backend) : Execution(backend) {
    mType = opType;
    mAxis = axis;
}
ReductionExecution::~ ReductionExecution() {
    // Do nothing
}

template <typename T>
__global__ void SUM(const T *input, T *output, int inside, int axis, int outside) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        T sumValue = (T)0;
        const T* basicInput = input + y * axis * inside + x;
        for (int v=0; v<axis; ++v) {
            sumValue += basicInput[v * inside];
        }
        output[y * inside + x] = sumValue;
    }
    return;
}

template <typename T>
__global__ void MEAN(const T *input, T *output, int inside, int axis, int outside) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        T sumValue = (T)0;
        const T* basicInput = input + y * axis * inside + x;
        for (int v=0; v<axis; ++v) {
            sumValue += basicInput[v * inside];
        }
        output[y * inside + x] = sumValue / (T)axis;
    }
    return;
}

template <typename T>
__global__ void MINIMUM(const T *input, T *output, int inside, int axis, int outside) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        const T* basicInput = input + y * axis * inside + x;
        T res = basicInput[0];
        for (int v=1; v<axis; ++v) {
            res = min(basicInput[v * inside], res);
        }
        output[y * inside + x] = res;
    }
    return;
}

template <typename T>
__global__ void MAXIMUM(const T *input, T *output, int inside, int axis, int outside) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        const T* basicInput = input + y * axis * inside + x;
        T res = basicInput[0];
        for (int v=1; v<axis; ++v) {
            res = max(basicInput[v * inside], res);
        }
        output[y * inside + x] = res;
    }
    return;
}

template <typename T>
__global__ void PROD(const T *input, T *output, int inside, int axis, int outside) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        const T* basicInput = input + y * axis * inside + x;
        T res = basicInput[0];
        for (int v=1; v<axis; ++v) {
            res *= basicInput[v * inside];
        }
        output[y * inside + x] = res;
    }
    return;
}

ErrorCode ReductionExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = (void*)inputs[0]->deviceId();
    auto output = (void*)outputs[0]->deviceId();
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    int inside = 1;
    int outside = 1;
    int axis = inputs[0]->length(mAxis);
    for (int i=0; i<mAxis; ++i) {
        outside *= inputs[0]->length(i);
    }
    for (int i=mAxis+1; i<inputs[0]->dimensions(); ++i) {
        inside *= inputs[0]->length(i);
    }
    int count = inside * outside;
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    if (inputs[0]->getType() == halide_type_of<float>()) {
        switch (mType) {
            case ReductionType_MEAN:
                MEAN<<<block_num, threads_num>>>((const float*)input, (float*)output, inside, axis, outside);
                return NO_ERROR;
            case ReductionType_SUM:
                SUM<<<block_num, threads_num>>>((const float*)input, (float*)output, inside, axis, outside);
                return NO_ERROR;
            case ReductionType_MINIMUM:
                MINIMUM<<<block_num, threads_num>>>((const float*)input, (float*)output, inside, axis, outside);
                return NO_ERROR;
            case ReductionType_MAXIMUM:
                MAXIMUM<<<block_num, threads_num>>>((const float*)input, (float*)output, inside, axis, outside);
                return NO_ERROR;
            case ReductionType_PROD:
                PROD<<<block_num, threads_num>>>((const float*)input, (float*)output, inside, axis, outside);
                return NO_ERROR;
        }
        MNN_ASSERT(false);
        return NOT_SUPPORT;
    }
    MNN_ASSERT(inputs[0]->getType() == halide_type_of<int32_t>());
    switch (mType) {
        case ReductionType_MEAN:
            MEAN<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, inside, axis, outside);
            return NO_ERROR;
        case ReductionType_SUM:
            SUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, inside, axis, outside);
            return NO_ERROR;
        case ReductionType_MINIMUM:
            MINIMUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, inside, axis, outside);
            return NO_ERROR;
        case ReductionType_MAXIMUM:
            MAXIMUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, inside, axis, outside);
            return NO_ERROR;
        case ReductionType_PROD:
            PROD<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, inside, axis, outside);
            return NO_ERROR;
        case ReductionType_ANY:
            MAXIMUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, inside, axis, outside);
            return NO_ERROR;
        case ReductionType_ALL:
            MINIMUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, inside, axis, outside);
            return NO_ERROR;
    }
    MNN_ASSERT(false);
    return NOT_SUPPORT;
}

class ReductionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto type = inputs[0]->getType();
        if (type.bits != 32) {
            return nullptr;
        }
        if (type.code != halide_type_float && type.code != halide_type_int) {
            return nullptr;
        }
        auto axis = op->main_as_ReductionParam()->dim()->data()[0];
        auto opType = op->main_as_ReductionParam()->operation();
        return new ReductionExecution(opType, axis, backend);
    }
};

static CUDACreatorRegister<ReductionCreator> __init(OpType_Reduction);


}
}
