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
#include "compute/CommonOptFunction.h"
#include "compute/StrassenMatmulComputor.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
struct DeconvolutionResource {
    struct Param {
        int outputCount;
        int srcCount;
        int fh;
        int fw;
    };
    Param mParam;
    std::shared_ptr<Tensor> mBias;
    std::shared_ptr<Tensor> mWeight;
};
class CPUDeconvolutionBasic : public CPUConvolution {
public:
    CPUDeconvolutionBasic(int inputChannel, const Op *convOp, Backend *b);
    virtual ~CPUDeconvolutionBasic() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    int mSrcCount;
    std::vector<float> mPostParameters;
};

class CPUDeconvolutionOrigin : public CPUDeconvolutionBasic {
public:
    CPUDeconvolutionOrigin(int inputChannel, const Op *convOp, Backend *b);
    virtual ~CPUDeconvolutionOrigin() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    MemChunk mGemmOutput;
    MemChunk mGemmInput;
    MemChunk mExtraOutput;

    std::vector<std::pair<std::function<void(uint8_t*, int)>, int>> mExecuteFuntion;
};

class CPUDeconvolution : public CPUDeconvolutionBasic {
public:
    static std::shared_ptr<DeconvolutionResource> makeResource(int inputChannel, const Op *convOp, Backend *b, bool dynamic);
    CPUDeconvolution(int inputChannel, const Op *convOp, Backend *b, bool dynamicWeight, std::shared_ptr<DeconvolutionResource> res);
    virtual ~CPUDeconvolution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    bool mDynamicWeight;
    std::shared_ptr<DeconvolutionResource> mResource;
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;
    std::shared_ptr<Tensor> mWeightTransformCache;
    std::vector<Tensor *> mTempInputs;
    std::shared_ptr<CPUDeconvolutionOrigin> mOrigin;
};
} // namespace MNN
#endif /* CPUDeconvolution_hpp */
