#include "ReductionExecution.hpp"
namespace MNN {
namespace CUDA {

template<typename T>
static void callSumFunc(const T* input, T* output, ReduceParam* param, CUDARuntime* runtime) {
    int inside  = param->inside;
    int outside = param->outside;
    int axis    = param->axis;
    int count = outside * inside;

    if(axis % 256 == 0 || axis >= 768) {
        int calc_multi_num = (axis + 255) / 256;
        SUM_REDUCE_AXIS<<<count, 256>>>(input, output, outside, axis, inside, 256, calc_multi_num);
        checkKernelErrors;
    } else if(axis >= 32) {
        int calc_multi_num = (axis + 63) / 64;
        SUM_REDUCE_AXIS<<<count, 64>>>(input, output, outside, axis, inside, 64, calc_multi_num);
        checkKernelErrors;
    } else {
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        SUM_NAIVE<<<block_num, threads_num>>>(input, output, outside, axis, inside);
        checkKernelErrors;
    }
}

template<typename T>
static void callMeanFunc(const T* input, T* output, ReduceParam* param, CUDARuntime* runtime) {
    int inside  = param->inside;
    int outside = param->outside;
    int axis    = param->axis;
    int count = outside * inside;

    if(axis % 256 == 0 || axis >= 768) {
        int calc_multi_num = (axis + 255) / 256;
        MEAN_REDUCE_AXIS<<<count, 256>>>(input, output, outside, axis, inside, 256, calc_multi_num);
        checkKernelErrors;
    } else if(axis >= 32) {
        int calc_multi_num = (axis + 63) / 64;
        MEAN_REDUCE_AXIS<<<count, 64>>>(input, output, outside, axis, inside, 64, calc_multi_num);
        checkKernelErrors;
    } else {
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        MEAN_NAIVE<<<block_num, threads_num>>>(input, output, outside, axis, inside);
        checkKernelErrors;
    }
}

template<typename T>
static void callMaxFunc(const T* input, T* output, ReduceParam* param, CUDARuntime* runtime) {
    int inside  = param->inside;
    int outside = param->outside;
    int axis    = param->axis;
    int count = outside * inside;

    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    MAXIMUM<<<block_num, threads_num>>>(input, output, outside, axis, inside);
    checkKernelErrors;
}

template<typename T>
static void callMinFunc(const T* input, T* output, ReduceParam* param, CUDARuntime* runtime) {
    int inside  = param->inside;
    int outside = param->outside;
    int axis    = param->axis;
    int count = outside * inside;

    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    MINIMUM<<<block_num, threads_num>>>(input, output, outside, axis, inside);
    checkKernelErrors;
}

template<typename T>
static void callProdFunc(const T* input, T* output, ReduceParam* param, CUDARuntime* runtime) {
    int inside  = param->inside;
    int outside = param->outside;
    int axis    = param->axis;
    int count = outside * inside;

    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    PROD<<<block_num, threads_num>>>(input, output, outside, axis, inside);
    checkKernelErrors;
}

ReductionExecution::ReductionExecution(ReductionType opType, int axis, Backend *backend) : Execution(backend) {
    mType = opType;
    mAxis = axis;
}
ReductionExecution::~ ReductionExecution() {
}

ErrorCode ReductionExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    mCpuParam.inside = inside;
    mCpuParam.outside = outside;
    mCpuParam.axis = axis;
    // MNN_PRINT("Reduction axis_idx:%d, outside:%d, axis:%d, inside:%d\n", mAxis, outside, axis, inside);
    return NO_ERROR;
}

ErrorCode ReductionExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = (void*)inputs[0]->deviceId();
    auto output = (void*)outputs[0]->deviceId();
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    int inside = mCpuParam.inside;
    int outside = mCpuParam.outside;
    int count = inside * outside;
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    if (inputs[0]->getType() == halide_type_of<float>()) {
        if (static_cast<CUDABackend*>(backend())->useFp16()) {
            switch (mType) {
                case ReductionType_MEAN:
                    callMeanFunc((const half*)input, (half*)output, &mCpuParam, runtime);
                    return NO_ERROR;
                case ReductionType_SUM:
                    callSumFunc((const half*)input, (half*)output, &mCpuParam, runtime);
                    return NO_ERROR;
                case ReductionType_MINIMUM:
                    callMinFunc((const half*)input, (half*)output, &mCpuParam, runtime);
                    return NO_ERROR;
                case ReductionType_MAXIMUM:
                    callMaxFunc((const half*)input, (half*)output, &mCpuParam, runtime);
                    return NO_ERROR;
                case ReductionType_PROD:
                    callProdFunc((const half*)input, (half*)output, &mCpuParam, runtime);
                    return NO_ERROR;
            }
        } else {
            switch (mType) {
                case ReductionType_MEAN:
                    callMeanFunc((const float*)input, (float*)output, &mCpuParam, runtime);
                    return NO_ERROR;
                case ReductionType_SUM:
                    callSumFunc((const float*)input, (float*)output, &mCpuParam, runtime);
                    return NO_ERROR;
                case ReductionType_MINIMUM:
                    callMinFunc((const float*)input, (float*)output, &mCpuParam, runtime);
                    return NO_ERROR;
                case ReductionType_MAXIMUM:
                    callMaxFunc((const float*)input, (float*)output, &mCpuParam, runtime);
                    return NO_ERROR;
                case ReductionType_PROD:
                    callProdFunc((const float*)input, (float*)output, &mCpuParam, runtime);
                    return NO_ERROR;
            }
        }
        MNN_ASSERT(false);
        return NOT_SUPPORT;
    }
    
    MNN_ASSERT(inputs[0]->getType() == halide_type_of<int32_t>());
    switch (mType) {
        case ReductionType_MEAN:
            callMeanFunc((const int32_t*)input, (int32_t*)output, &mCpuParam, runtime);
            return NO_ERROR;
        case ReductionType_SUM:
            callSumFunc((const int32_t*)input, (int32_t*)output, &mCpuParam, runtime);
            // SUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, param);
            return NO_ERROR;
        case ReductionType_MINIMUM:
            callMinFunc((const int32_t*)input, (int32_t*)output, &mCpuParam, runtime);
            return NO_ERROR;
        case ReductionType_MAXIMUM:
            callMaxFunc((const int32_t*)input, (int32_t*)output, &mCpuParam, runtime);
            return NO_ERROR;
        case ReductionType_PROD:
            callProdFunc((const int32_t*)input, (int32_t*)output, &mCpuParam, runtime);
            return NO_ERROR;
        case ReductionType_ANY:
            callMaxFunc((const int32_t*)input, (int32_t*)output, &mCpuParam, runtime);
            return NO_ERROR;
        case ReductionType_ALL:
            callMinFunc((const int32_t*)input, (int32_t*)output, &mCpuParam, runtime);
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
