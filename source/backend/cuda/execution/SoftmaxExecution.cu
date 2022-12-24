#include "SoftmaxExecution.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
namespace CUDA {

template <typename T>
__global__ void SOFTMAX(const T *input, T *output,
    const int inside,
    const int axis,
    const int outside,
    const int count
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        const T* src = input + y * axis * inside + x;
        T* dst = output + y * axis * inside + x;
        float maxValue = (float)src[0];
        for (int z=1; z<axis; ++z) {
            maxValue = max(maxValue, src[z * inside]);
        }
        float sumValue = 0.0;
        for (int z=0; z<axis; ++z) {
            sumValue = sumValue + exp((float)src[z * inside] - maxValue);
        }
        sumValue = 1.0 / sumValue;
        for (int z=0; z<axis; ++z) {
            dst[z*inside] = (T)(exp((float)src[z * inside] - maxValue) * sumValue);
        }
    }
}

template <typename T>
__global__ void EXPSUB(const T *input, const T* maxV, T *output, 
    const int inside,
    const int axis,
    const int outside,
    const int count
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int tmp = i / inside;
        int x = i % inside;
        int y = tmp / axis;
        int c = tmp % axis;
        float sumValue = 0.0;
        const float basicInput = input[i];
        const float maxValue = maxV[x + y * inside];
        output[i] = (T)(exp(basicInput - maxValue));
    }
    return;
}

template <typename T>
__global__ void DIVSUM(const T *input, const T* maxV, T *output,
    const int inside,
    const int axis,
    const int outside,
    const int count
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int tmp = i / inside;
        int x = i % inside;
        int y = tmp / axis;
        int c = tmp % axis;
        float sumValue = 0.0;
        const float basicInput = input[i];
        const float value = maxV[x + y * inside];
        output[i] = (T)(basicInput / value);
    }
    return;
}
SoftmaxExecution::SoftmaxExecution(int axis, Backend *backend) : Execution(backend) {
    mAxis = axis;
}

SoftmaxExecution::~SoftmaxExecution() {
    //
}

ErrorCode SoftmaxExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input           = inputs[0];
    const int dimensions = input->buffer().dimensions;
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    int axis = mAxis;
    if (axis < 0) {
        axis += dimensions;
    }

    const auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
    mNeedUnpackC4     = layout == MNN_DATA_FORMAT_NC4HW4;
    if (mNeedUnpackC4) {    
        TensorUtils::copyShape(input, &mStorage);
        TensorUtils::getDescribe(&mStorage)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        mStorage.buffer().dimensions    = dimensions;
        mStorage.buffer().type          = input->getType();
        backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);
    }

    int inside = 1;
    int outside = 1;
    int dims   = input->buffer().dimensions;
    for (int i = 0; i < axis; ++i) {
        outside *= input->length(i);
    }
    for (int i = axis + 1; i < dims; ++i) {
        inside *= input->length(i);
    }

    if (mNeedUnpackC4) {
        backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);
    }

    mCpuParam.inside = inside;
    mCpuParam.outside = outside;
    mCpuParam.axis = input->length(axis);

    return NO_ERROR;
}

ErrorCode SoftmaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = (void*)inputs[0]->deviceId();
    auto output = (void*)outputs[0]->deviceId();
    auto dst = output;

    if (mNeedUnpackC4) {
        backend()->onCopyBuffer(inputs[0], &mStorage);
        input = (void*)mStorage.deviceId();
        dst = (void*)mStorage.deviceId();
    }

    //MNN_PRINT("softmax input dims:%d, size:%d-%d-%d-%d\n", inputs[0]->dimensions(), inputs[0]->batch(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel());
    //MNN_PRINT("softmax storage dims:%d, size:%d-%d-%d-%d\n", mStorage.dimensions(), mStorage.batch(), mStorage.height(), mStorage.width(), mStorage.channel());

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    int inside = mCpuParam.inside;
    int outside = mCpuParam.outside;
    int axis = mCpuParam.axis;
    int count = inside * outside;
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        SOFTMAX<<<block_num, threads_num>>>((const half*)input, (half*)dst, inside, axis, outside, count);
    } else {
        SOFTMAX<<<block_num, threads_num>>>((const float*)input, (float*)dst, inside, axis, outside, count);
    }
    if (mNeedUnpackC4) {
        backend()->onCopyBuffer(&mStorage, outputs[0]);
    }

    return NO_ERROR;
}

class SoftmaxCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto type = inputs[0]->getType();
        if (type.code != halide_type_float) {
            MNN_PRINT("softmax data type:%s not support", type.code);
            return nullptr;
        }
        auto axis = op->main_as_Axis()->axis();
        return new SoftmaxExecution(axis, backend);
    }
};

static CUDACreatorRegister<SoftmaxCreator> __init(OpType_Softmax);
}
}