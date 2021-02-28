#include "SoftmaxExecution.hpp"

namespace MNN {
namespace CUDA {

SoftmaxExecution::SoftmaxExecution(int axis, Backend *backend) : Execution(backend) {
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    cudnn_handle_ = runtime->cudnn_handle();

    cudnn_check(cudnnCreateTensorDescriptor(&input_desc_));
    cudnn_check(cudnnCreateTensorDescriptor(&output_desc_));

    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    mAxis = axis;
}

SoftmaxExecution::~SoftmaxExecution() {
    cudnnDestroyTensorDescriptor(input_desc_);
    cudnnDestroyTensorDescriptor(output_desc_);
}

ErrorCode SoftmaxExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    inside = 1;
    outside = 1;
    if(mAxis < 0) {
        mAxis += inputs[0]->dimensions();
    }
    axis = inputs[0]->length(mAxis);
    for (int i=0; i<mAxis; ++i) {
        outside *= inputs[0]->length(i);
    }
    for (int i=mAxis+1; i<inputs[0]->dimensions(); ++i) {
        inside *= inputs[0]->length(i);
    }

    std::vector<int> tensor_shape = {outside, axis, inside, 1};
    cudnn_check(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, tensor_shape[0],
                                tensor_shape[1], tensor_shape[2], tensor_shape[3]));

    cudnn_check(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, tensor_shape[0],
                                tensor_shape[1], tensor_shape[2], tensor_shape[3]));

    return NO_ERROR;
}

ErrorCode SoftmaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = (void*)inputs[0]->deviceId();
    auto output = (void*)outputs[0]->deviceId();

    const float alpha = 1;
    const float beta = 0;
    cudnn_check(cudnnSoftmaxForward(cudnn_handle_, CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha,
                input_desc_, input,
                &beta,
                output_desc_, output));

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