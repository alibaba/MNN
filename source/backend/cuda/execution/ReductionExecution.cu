#include "ReductionExecution.hpp"
namespace MNN {
namespace CUDA {

ReductionExecution::ReductionExecution(ReductionType opType, int axis, Backend *backend) : Execution(backend) {
    mType = opType;
    mAxis = axis;
    auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
    mParam = staticPool->alloc(sizeof(ReduceParam));
}
ReductionExecution::~ ReductionExecution() {
    auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    staticPool->free(mParam);
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
    cuda_check(cudaMemcpy((uint8_t*)mParam.first + mParam.second, &mCpuParam, sizeof(ReduceParam), cudaMemcpyHostToDevice));
    
    //MNN_PRINT("Reduction axis_idx:%d, outside:%d, axis:%d, inside:%d\n", mAxis, outside, axis, inside);
    return NO_ERROR;
}

ErrorCode ReductionExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = (void*)inputs[0]->deviceId();
    auto output = (void*)outputs[0]->deviceId();
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    int inside = mCpuParam.inside;;
    int outside = mCpuParam.outside;
    int count = inside * outside;
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    auto param = (ReduceParam*)((uint8_t*)mParam.first + mParam.second);
    if (inputs[0]->getType() == halide_type_of<float>()) {
        if (static_cast<CUDABackend*>(backend())->useFp16()) {
            switch (mType) {
                case ReductionType_MEAN:
                    MEAN<<<block_num, threads_num>>>((const half*)input, (half*)output, param);
                    return NO_ERROR;
                case ReductionType_SUM:
                    SUM<<<block_num, threads_num>>>((const half*)input, (half*)output, param);
                    return NO_ERROR;
                case ReductionType_MINIMUM:
                    MINIMUM<<<block_num, threads_num>>>((const half*)input, (half*)output, param);
                    return NO_ERROR;
                case ReductionType_MAXIMUM:
                    MAXIMUM<<<block_num, threads_num>>>((const half*)input, (half*)output, param);
                    return NO_ERROR;
                case ReductionType_PROD:
                    PROD<<<block_num, threads_num>>>((const half*)input, (half*)output, param);
                    return NO_ERROR;
            }
        } else {
            switch (mType) {
                case ReductionType_MEAN:
                    MEAN<<<block_num, threads_num>>>((const float*)input, (float*)output, param);
                    return NO_ERROR;
                case ReductionType_SUM:
                    SUM<<<block_num, threads_num>>>((const float*)input, (float*)output, param);
                    return NO_ERROR;
                case ReductionType_MINIMUM:
                    MINIMUM<<<block_num, threads_num>>>((const float*)input, (float*)output, param);
                    return NO_ERROR;
                case ReductionType_MAXIMUM:
                    MAXIMUM<<<block_num, threads_num>>>((const float*)input, (float*)output, param);
                    return NO_ERROR;
                case ReductionType_PROD:
                    PROD<<<block_num, threads_num>>>((const float*)input, (float*)output, param);
                    return NO_ERROR;
            }
        }
        MNN_ASSERT(false);
        return NOT_SUPPORT;
    }
    
    MNN_ASSERT(inputs[0]->getType() == halide_type_of<int32_t>());
    switch (mType) {
        case ReductionType_MEAN:
            MEAN<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, param);
            return NO_ERROR;
        case ReductionType_SUM:
            SUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, param);
            return NO_ERROR;
        case ReductionType_MINIMUM:
            MINIMUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, param);
            return NO_ERROR;
        case ReductionType_MAXIMUM:
            MAXIMUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, param);
            return NO_ERROR;
        case ReductionType_PROD:
            PROD<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, param);
            return NO_ERROR;
        case ReductionType_ANY:
            MAXIMUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, param);
            return NO_ERROR;
        case ReductionType_ALL:
            MINIMUM<<<block_num, threads_num>>>((const int32_t*)input, (int32_t*)output, param);
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
