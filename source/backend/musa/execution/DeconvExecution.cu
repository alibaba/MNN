#include "DeconvExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void Deconv2dKernel(const T* input, const T* weight, T* output,
                                int batch, int inChannels, int outChannels,
                                int inHeight, int inWidth,
                                int outHeight, int outWidth,
                                int kernelH, int kernelW,
                                int strideH, int strideW,
                                int padH, int padW,
                                int dilationH, int dilationW,
                                int group) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batch * outChannels * outHeight * outWidth;
    
    if (index < totalSize) {
        int tmp = index;
        int outW = tmp % outWidth;
        tmp /= outWidth;
        int outH = tmp % outHeight;
        tmp /= outHeight;
        int outC = tmp % outChannels;
        int b = tmp / outChannels;
        
        int inCBase = (outC / (outChannels / group)) * (inChannels / group);
        int channelPerGroup = outChannels / group;
        
        T sum = 0;
        for (int ic = 0; ic < inChannels / group; ic++) {
            int inC = inCBase + ic;
            for (int kh = 0; kh < kernelH; kh++) {
                for (int kw = 0; kw < kernelW; kw++) {
                    int inH = outH * strideH + kh * dilationH - padH;
                    int inW = outW * strideW + kw * dilationW - padW;
                    
                    if (inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth) {
                        int inIdx = ((b * inChannels + inC) * inHeight + inH) * inWidth + inW;
                        int wIdx = ((outC * (inChannels / group) + ic) * kernelH + kh) * kernelW + kw;
                        sum += input[inIdx] * weight[wIdx];
                    }
                }
            }
        }
        output[index] = sum;
    }
}

DeconvExecution::DeconvExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_Convolution2D();
}

ErrorCode DeconvExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    mBatch = input->batch();
    mInChannels = input->channel();
    mOutChannels = output->channel();
    mInHeight = input->height();
    mInWidth = input->width();
    mOutHeight = output->height();
    mOutWidth = output->width();
    
    auto common = mOp->common();
    mKernelH = common->kernelY();
    mKernelW = common->kernelX();
    mStrideH = common->strideY();
    mStrideW = common->strideX();
    mPadH = common->padY();
    mPadW = common->padX();
    mDilationH = common->dilatedY();
    mDilationW = common->dilatedX();
    mGroup = common->group();
    
    int threads = 256;
    int totalSize = mBatch * mOutChannels * mOutHeight * mOutWidth;
    int blocks = (totalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode DeconvExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto weight = inputs[1];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto weightPtr = weight->host<float>();
    auto outputPtr = output->host<float>();
    
    Deconv2dKernel<<<mDim3Grid, mDim3Block>>>(
        inputPtr, weightPtr, outputPtr,
        mBatch, mInChannels, mOutChannels,
        mInHeight, mInWidth,
        mOutHeight, mOutWidth,
        mKernelH, mKernelW,
        mStrideH, mStrideW,
        mPadH, mPadW,
        mDilationH, mDilationW,
        mGroup
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class DeconvCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new DeconvExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<DeconvCreator> gDeconvRegistration(OpType_Deconvolution);

} // namespace MUSA
} // namespace MNN
