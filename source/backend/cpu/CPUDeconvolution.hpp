//
//  CPUDeconvolution.hpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDeconvolution_hpp
#define CPUDeconvolution_hpp

#include "CPUConvolution.hpp"
#include "compute/StrassenMatmulComputor.hpp"

namespace MNN {
class CPUDeconvolutionBasic : public CPUConvolution {
public:
    CPUDeconvolutionBasic(const Tensor *input, const Op *convOp, Backend *b);
    virtual ~CPUDeconvolutionBasic() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    int mSrcCount;
};

class CPUDeconvolutionCommon : public CPUDeconvolutionBasic {
public:
    CPUDeconvolutionCommon(const Tensor *input, const Op *convOp, Backend *b);
    virtual ~CPUDeconvolutionCommon();

protected:
    std::shared_ptr<Tensor> mBias;
};

class CPUDeconvolutionOrigin : public CPUDeconvolutionBasic {
public:
    CPUDeconvolutionOrigin(const Tensor *input, const Op *convOp, Backend *b)
        : CPUDeconvolutionBasic(input, convOp, b) {
        // Do nothing
    }
    virtual ~CPUDeconvolutionOrigin() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<StrassenMatrixComputor> mMatMul;
    std::vector<std::pair<std::function<void(int)>, int>> mPreFunctions;
    std::vector<std::pair<std::function<void(int)>, int>> mPostFunctions;
};
class CPUDeconvolutionMultiInput : public CPUDeconvolutionBasic {
public:
    CPUDeconvolutionMultiInput(const Tensor *input, const Op *convOp, Backend *b)
        : CPUDeconvolutionBasic(input, convOp, b) {
        mOrigin.reset(new CPUDeconvolutionOrigin(input, convOp, b));
    }
    virtual ~CPUDeconvolutionMultiInput() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mCacheWeight;
    std::shared_ptr<Tensor> mBias;
    std::vector<Tensor *> mTempInputs;
    std::shared_ptr<CPUDeconvolutionOrigin> mOrigin;
};

class CPUDeconvolution : public CPUDeconvolutionCommon {
public:
    CPUDeconvolution(const Tensor *input, const Op *convOp, Backend *b);
    virtual ~CPUDeconvolution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        mOrigin->onExecute(mTempInputs, outputs);
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        mTempInputs = {inputs[0], mWeight.get(), mBias.get()};
        return mOrigin->onResize(mTempInputs, outputs);
    }

private:
    std::shared_ptr<Tensor> mWeight;
    std::vector<Tensor *> mTempInputs;
    std::shared_ptr<CPUDeconvolutionOrigin> mOrigin;
};
} // namespace MNN
#endif /* CPUDeconvolution_hpp */
