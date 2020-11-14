//
//  DeconvSingleInputExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DeconvSingleInputExecution_hpp
#define DeconvSingleInputExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "half.hpp"

namespace MNN {
namespace CUDA {

struct IOInfo {
    int ib, ic, ih, iw;
    int ob, oc, oh, ow;
};
struct KernelInfo {
    int groups         = 0;
    int kernelN        = 0;
    int kernelC        = 0;
    int kernelX        = 0;
    int kernelY        = 0;
    PadMode padMode    = PadMode_CAFFE;
    int padX           = 0;
    int padY           = 0;
    int strideX        = 0;
    int strideY        = 0;
    int dilateX        = 0;
    int dilateY        = 0;
    int activationType = 0;
};//

extern "C"
class DeconvSingleInputExecution : public Execution {
public:
    DeconvSingleInputExecution(Backend* backend, const MNN::Op* op);
    virtual ~DeconvSingleInputExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;

private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionBwdDataAlgo_t conv_bwd_algo_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    cudnnTensorDescriptor_t padded_desc_;
    cudnnActivationDescriptor_t act_desc_;

    cudnnDataType_t cudnn_data_type_;
    int cudnn_data_type_len_;
    bool use_pad_ = false;
    int pad_top_ = 0;
    int pad_bottom_ = 0;
    int pad_left_ = 0;
    int pad_right_ = 0;

    bool use_bias_ = false;
    bool use_relu_ = false;
    bool use_relu6_ = false;

    void* mPadPtr;
    void* mFilter;
    void* mBias;
    void* mWorkSpace;
    std::shared_ptr<Tensor> weightTensor;
    std::shared_ptr<Tensor> biasTensor;
    std::shared_ptr<Tensor> padTensor;
    std::shared_ptr<Tensor> workspaceTensor;

    std::shared_ptr<Tensor> mPad;
    std::shared_ptr<Tensor> mWorkspaceForward;

    size_t input_size_;
    size_t filter_size_;
    size_t output_size_;
    size_t padded_size_;
    size_t workspace_size_;

    const MNN::Op* mOp;
    KernelInfo mKernelInfo;
    IOInfo mIOInfo;
    std::shared_ptr<Tensor> mTempInput;
};

} // namespace CUDA
} // namespace MNN

#endif /* DeconvSingleInputExecution */
