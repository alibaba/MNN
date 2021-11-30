//
//  ConvSingleInputExecution.cpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvSingleInputExecution.hpp"

namespace MNN {
namespace CUDA {

template <typename T>
__global__ void Pad(const size_t size, const T* input, const int old_height,
                    const int old_width, const int padded_height, const int padded_width, const int pad_top,
                    const int pad_left, float pad_value, T* output) {
    T pad_value_ = static_cast<T>(pad_value);
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
        int block_num = pos / (padded_width*padded_height);
        int left = pos % (padded_width*padded_height);
        const int padded_w = left % padded_width;
        const int padded_h = left / padded_width % padded_height;
        if (padded_h - pad_top < 0 || padded_w - pad_left < 0 || padded_h - pad_top >= old_height ||
              padded_w - pad_left >= old_width) {
            output[pos] = pad_value_;
        } else {
            output[pos] = input[(block_num * old_height + padded_h - pad_top) * old_width + padded_w - pad_left];
        }
    }
    return;
}

ConvSingleInputExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    cudnn_data_type_len_ = 0;
    mKernelInfo.kernelX        = common->kernelX();
    mKernelInfo.kernelY        = common->kernelY();
    mKernelInfo.groups         = common->group();
    mKernelInfo.padMode        = common->padMode();
    mKernelInfo.padX           = common->padX();
    mKernelInfo.padY           = common->padY();

    if (nullptr != common->pads()) {
        mKernelInfo.padX = common->pads()->data()[1];
        mKernelInfo.padY = common->pads()->data()[0];
    }
    mKernelInfo.strideX        = common->strideX();
    mKernelInfo.strideY        = common->strideY();
    mKernelInfo.dilateX        = common->dilateX();
    mKernelInfo.dilateY        = common->dilateY();
    mKernelInfo.activationType = common->relu() ? 1 : (common->relu6() ? 2 : 0);
    use_relu_ = (mKernelInfo.activationType == 1);
    use_relu6_ = (mKernelInfo.activationType == 2);
    use_bias_ = true;
    cudnn_check(cudnnCreateActivationDescriptor(&act_desc_));
    cudnn_check(cudnnCreateTensorDescriptor(&bias_desc_));
    cudnn_check(cudnnCreateFilterDescriptor(&filter_desc_));
    cudnn_check(cudnnCreateConvolutionDescriptor(&conv_desc_));

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, conv, &filterDataPtr, &weightSize);
    mKernelInfo.kernelN = common->outputCount();
    mKernelInfo.kernelC = weightSize / mKernelInfo.kernelN / mKernelInfo.kernelX / mKernelInfo.kernelY;

    weightTensor.reset(Tensor::createDevice<float>({weightSize}));
    bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
    mFilter = (void *)weightTensor.get()->buffer().device;
    cuda_check(cudaMemcpy(mFilter, filterDataPtr, weightSize*sizeof(float), cudaMemcpyHostToDevice));

    int biasSize = conv->bias()->size();
    biasTensor.reset(Tensor::createDevice<float>({biasSize}));
    bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
    mBias = (void *)biasTensor.get()->buffer().device;

    cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

    int bias_size = conv->bias()->size();
    int dim_bias[] = {1, bias_size, 1, 1};
    int stride_bias[] = {bias_size, 1, 1, 1};
    if(cudnn_data_type_ == CUDNN_DATA_FLOAT) {
        cudnn_check(cudnnSetTensorNdDescriptor(bias_desc_, CUDNN_DATA_FLOAT, 4, dim_bias, stride_bias));
    }
    else if(cudnn_data_type_ == CUDNN_DATA_HALF) {
        cudnn_check(cudnnSetTensorNdDescriptor(bias_desc_, CUDNN_DATA_HALF, 4, dim_bias, stride_bias));
    } else {
        MNN_PRINT("only supports fp32/fp16 data type!!!\n");
    }
    use_bias_ = true;

    mKernelInfo.kernelN = common->outputCount();
    mKernelInfo.kernelC = weightSize / (mKernelInfo.kernelN * mKernelInfo.kernelY * mKernelInfo.kernelX);
    std::vector<int> filter_shape = {mKernelInfo.kernelN, mKernelInfo.kernelC, mKernelInfo.kernelY, mKernelInfo.kernelX};

    cudnn_check(cudnnSetFilter4dDescriptor(filter_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, filter_shape[0],
        filter_shape[1], filter_shape[2], filter_shape[3]));
    cudnn_check(cudnnSetConvolution2dDescriptor(conv_desc_, 0, 0, mKernelInfo.strideY, mKernelInfo.strideX, 
            mKernelInfo.dilateY, mKernelInfo.dilateX, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
        cudnn_check(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));
    }
    //set group num
    cudnn_check(cudnnSetConvolutionGroupCount(conv_desc_, mKernelInfo.groups));
    if(use_relu_) {
        cudnn_check(cudnnSetActivationDescriptor(act_desc_, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    } else if(use_relu6_) {
        cudnn_check(cudnnSetActivationDescriptor(act_desc_, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_NOT_PROPAGATE_NAN, 6.0));
    } else {
        //do nothing
    }
}

ConvSingleInputExecution::Resource::~Resource() {
    cudnn_check(cudnnDestroyFilterDescriptor(filter_desc_));
    cudnn_check(cudnnDestroyTensorDescriptor(bias_desc_));
    cudnn_check(cudnnDestroyActivationDescriptor(act_desc_));
    cudnn_check(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

ConvSingleInputExecution::ConvSingleInputExecution(Backend* backend, const MNN::Op* op) : Execution(backend), mOp(op) {
    //MNN_PRINT("cuda convSingleInput onInit in\n");
    mResource.reset(new Resource(backend, op));
    cudnn_handle_ = nullptr;
    input_desc_ = nullptr;
    output_desc_ = nullptr;
    padded_desc_ = nullptr;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    cudnn_handle_ = runtime->cudnn_handle();
    cudnn_check(cudnnCreateTensorDescriptor(&input_desc_));
    cudnn_check(cudnnCreateTensorDescriptor(&output_desc_));
    cudnn_check(cudnnCreateTensorDescriptor(&padded_desc_));
}
ConvSingleInputExecution::ConvSingleInputExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : Execution(backend), mOp(op) {
    mResource = res;
    cudnn_handle_ = nullptr;
    input_desc_ = nullptr;
    output_desc_ = nullptr;
    padded_desc_ = nullptr;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    cudnn_handle_ = runtime->cudnn_handle();
    cudnn_check(cudnnCreateTensorDescriptor(&input_desc_));
    cudnn_check(cudnnCreateTensorDescriptor(&output_desc_));
    cudnn_check(cudnnCreateTensorDescriptor(&padded_desc_));
}

ConvSingleInputExecution::~ConvSingleInputExecution() {
    cudnn_check(cudnnDestroyTensorDescriptor(padded_desc_));
    cudnn_check(cudnnDestroyTensorDescriptor(output_desc_));
    cudnn_check(cudnnDestroyTensorDescriptor(input_desc_));
}
bool ConvSingleInputExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvSingleInputExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}

ErrorCode ConvSingleInputExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    // prepare
    //MNN_PRINT("cuda convSingleInput onResize in, pad:%d\n", mKernelInfo.padX);
    auto input = inputs[0], output = outputs[0];

    mIOInfo.iw = input->width();
    mIOInfo.ih = input->height();
    mIOInfo.ic = input->channel();
    mIOInfo.ib = input->batch();
    
    mIOInfo.ow = output->width();
    mIOInfo.oh = output->height();
    mIOInfo.oc = output->channel();
    mIOInfo.ob = output->batch();

    if(mIOInfo.iw==0) {
        mIOInfo.iw = 1;
    }
    if(mIOInfo.ih==0) {
        mIOInfo.ih = 1;
    }
    if(mIOInfo.ic==0) {
        mIOInfo.ic = 1;
    }
    if(mIOInfo.ib==0) {
        mIOInfo.ib = 1;
    }
    if(mIOInfo.ow==0) {
        mIOInfo.ow = 1;
    }
    if(mIOInfo.oh==0) {
        mIOInfo.oh = 1;
    }
    if(mIOInfo.oc==0) {
        mIOInfo.oc = 1;
    }
    if(mIOInfo.ob==0) {
        mIOInfo.ob = 1;
    }
    std::vector<int> in_shape = {mIOInfo.ib, mIOInfo.ic, mIOInfo.ih, mIOInfo.iw};
    std::vector<int> output_shape = {mIOInfo.ob, mIOInfo.oc, mIOInfo.oh, mIOInfo.ow};
    auto cudnn_data_type_ = mResource->cudnn_data_type_;
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    pad_left_ = std::get<0>(pads);
    pad_top_ = std::get<1>(pads);
    pad_right_ = std::get<2>(pads);
    pad_bottom_ = std::get<3>(pads);
    // printf("filter:%d %d %d %d\n", filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]);
    // printf("input:%d %d %d %d\n", in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    // printf("output:%d %d %d %d\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    cudnn_check(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, in_shape[0],
                                in_shape[1], in_shape[2], in_shape[3]));

    cudnn_check(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, output_shape[0],
                                output_shape[1], output_shape[2], output_shape[3]));

    cudnnTensorDescriptor_t input_descriptor_real = nullptr;
    use_pad_ = (pad_left_!=0 || pad_right_!=0 || pad_top_!=0 || pad_bottom_!=0 ) ? true : false;

    if(use_pad_) {
        int totalSize = in_shape[0]*in_shape[1]*(in_shape[2]+pad_top_+pad_bottom_)*(in_shape[3]+pad_left_+pad_right_);
        padTensor.reset(Tensor::createDevice<float>({totalSize}));
        backend()->onAcquireBuffer(padTensor.get(), Backend::DYNAMIC);
        mPadPtr = (void *)padTensor.get()->buffer().device;

        //dynamic memory release
        backend()->onReleaseBuffer(padTensor.get(), Backend::DYNAMIC);

        cudnn_check(cudnnSetTensor4dDescriptor(padded_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, in_shape[0], in_shape[1],
                                in_shape[2] + +pad_top_+pad_bottom_, in_shape[3] + pad_left_+pad_right_));
    }
    input_descriptor_real = use_pad_ ? padded_desc_ : input_desc_;

    // algorithm
    constexpr int requested_algo_count = 1;
    int returned_algo_count;
    cudnnConvolutionFwdAlgoPerf_t perf_results;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle_, input_descriptor_real, mResource->filter_desc_, mResource->conv_desc_,
                                                output_desc_, requested_algo_count, &returned_algo_count, &perf_results));
    conv_algorithm_ = perf_results.algo;
    auto& mKernelInfo = mResource->mKernelInfo;

    if(mIOInfo.iw==1 && mIOInfo.ih==1 && mKernelInfo.kernelY==1 && mKernelInfo.kernelX==1) {
        conv_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    }
    // workspace
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, input_descriptor_real, mResource->filter_desc_, mResource->conv_desc_, output_desc_,
                                            conv_algorithm_, &workspace_size_));

    if (workspace_size_ != 0) {
        int workspaceSize = workspace_size_;
        workspaceTensor.reset(Tensor::createDevice<float>({workspaceSize}));
        //cudnn not support workspace memory reuse
        backend()->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
        mWorkSpace = (void *)workspaceTensor.get()->buffer().device;
    }
    //MNN_PRINT("cuda convSingleInput onResize out\n");
    return NO_ERROR;
}

ErrorCode ConvSingleInputExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    //MNN_PRINT("cuda convSingleInput onExecute in, inputsize:%d %d\n", (int)inputs.size(), workspace_size_);
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;

    void *output_addr = (void*)outputs[0]->deviceId();
    void *workspace_addr = nullptr;
    if (workspace_size_ != 0) {
        workspace_addr = mWorkSpace;
    }

    const float alpha = 1;
    const float beta = 0;

    if(use_pad_) {
        std::vector<int> in_shape = {mIOInfo.ib, mIOInfo.ic, mIOInfo.ih, mIOInfo.iw};

        int size = in_shape[0] * in_shape[1] * (in_shape[2]+pad_top_+pad_bottom_) * (in_shape[3]+pad_left_+pad_right_);
        int block_num = runtime->blocks_num(size);
        int threads_num = runtime->threads_num();

        Pad<<<block_num, threads_num>>>(size, (float*)input_addr, in_shape[2], in_shape[3],
            in_shape[2]+pad_top_+pad_bottom_, in_shape[3]+pad_left_+pad_right_, pad_top_, pad_left_, 0.0, (float*)mPadPtr);

        cudnn_check(cudnnConvolutionForward(cudnn_handle_, &alpha, padded_desc_, mPadPtr, mResource->filter_desc_, filter_addr, mResource->conv_desc_,
            conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr));
    }
    else {
        cudnn_check(cudnnConvolutionForward(cudnn_handle_, &alpha, input_desc_, input_addr, mResource->filter_desc_, filter_addr, mResource->conv_desc_,
            conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr));
    }

    if(mResource->use_bias_) {
        cudnn_check(cudnnAddTensor(cudnn_handle_, &alpha, mResource->bias_desc_, bias_addr, &alpha, output_desc_, output_addr));
    }
    if(mResource->use_relu_ || mResource->use_relu6_) {
        cudnn_check(cudnnActivationForward(cudnn_handle_, mResource->act_desc_, &alpha, output_desc_, output_addr, &beta, output_desc_, output_addr));
    }
    
    return NO_ERROR;
}

class CUDAConvolutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, 
            const MNN::Op* op, Backend* backend) const override {
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }
        return new ConvSingleInputExecution(backend, op);
    }
};

CUDACreatorRegister<CUDAConvolutionCreator> __ConvExecution(OpType_Convolution);

}// namespace CUDA
}// namespace MNN
