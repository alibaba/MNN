//
//  CPUDeconvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDeconvolutionDepthwise_hpp
#define CPUDeconvolutionDepthwise_hpp

#include "backend/cpu/CPUDeconvolution.hpp"

namespace MNN {
class CPUDeconvolutionDepthwiseBasic : public CPUDeconvolutionBasic {
public:
    CPUDeconvolutionDepthwiseBasic(int inputChannel, const Op *convOp, Backend *b)
        : CPUDeconvolutionBasic(inputChannel, convOp, b) {
    }
    virtual ~CPUDeconvolutionDepthwiseBasic() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::function<void(const uint8_t*, uint8_t*, int)> mFunction;
};

class CPUDeconvolutionDepthwiseMultiInput : public CPUDeconvolutionDepthwiseBasic {
public:
    CPUDeconvolutionDepthwiseMultiInput(int inputChannel, const Op *convOp, Backend *b)
        : CPUDeconvolutionDepthwiseBasic(inputChannel, convOp, b) {
    }
    virtual ~CPUDeconvolutionDepthwiseMultiInput() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;
    std::vector<Tensor *> mInputs;
};

class CPUDeconvolutionDepthwise : public CPUDeconvolutionBasic {
public:
    static std::shared_ptr<DeconvolutionResource> makeResource(int inputChannel, const Op *convOp, Backend *b);
    CPUDeconvolutionDepthwise(int inputChannel, const Op *convOp, Backend *b, std::shared_ptr<DeconvolutionResource> res);
    virtual ~CPUDeconvolutionDepthwise();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
        return mOrigin->onResize(mInputs, outputs);
    }
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return mOrigin->onExecute(mInputs, outputs);
    }

private:
    std::shared_ptr<DeconvolutionResource> mResource;
    std::vector<Tensor *> mInputs;
    std::unique_ptr<CPUDeconvolutionDepthwiseBasic> mOrigin;
};
} // namespace MNN

#endif /* CPUDeconvolutionDepthwise_hpp */
