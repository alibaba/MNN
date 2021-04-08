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
    CPUDeconvolutionDepthwiseBasic(const Tensor *input, const Op *convOp, Backend *b)
        : CPUDeconvolutionBasic(input, convOp, b) {
    }
    virtual ~CPUDeconvolutionDepthwiseBasic() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::function<void(const uint8_t*, uint8_t*, int)> mFunction;
};

class CPUDeconvolutionDepthwiseMultiInput : public CPUDeconvolutionDepthwiseBasic {
public:
    CPUDeconvolutionDepthwiseMultiInput(const Tensor *input, const Op *convOp, Backend *b)
        : CPUDeconvolutionDepthwiseBasic(input, convOp, b) {
    }
    virtual ~CPUDeconvolutionDepthwiseMultiInput() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;
    std::vector<Tensor *> mInputs;
};

class CPUDeconvolutionDepthwise : public CPUDeconvolutionCommon {
public:
    CPUDeconvolutionDepthwise(const Tensor *input, const Op *convOp, Backend *b);
    virtual ~CPUDeconvolutionDepthwise();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        mInputs = {inputs[0], mWeight.get(), mBias.get()};
        return mOrigin->onResize(mInputs, outputs);
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return mOrigin->onExecute(mInputs, outputs);
    }

private:
    std::shared_ptr<Tensor> mWeight;
    std::vector<Tensor *> mInputs;
    std::unique_ptr<CPUDeconvolutionDepthwiseBasic> mOrigin;
};
} // namespace MNN

#endif /* CPUDeconvolutionDepthwise_hpp */
