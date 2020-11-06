//
//  CPUConvInt8.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConvInt8_hpp
#define CPUConvInt8_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "compute/Int8FunctionsOpt.h"

namespace MNN {

class CPUConvInt8 : public CPUConvolution {
public:
    CPUConvInt8(Backend *backend, const MNN::Convolution2D *convOp, const std::vector<Tensor *> &inputs);
    virtual ~CPUConvInt8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    // relu or relu6
    bool mRelu;
    int mActBits;

    std::shared_ptr<Tensor> mWeightInt8;
    std::shared_ptr<Tensor> mBiasInt32;
    std::shared_ptr<Tensor> mScaleFloat;

    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;

    Tensor mTempIm2ColBuffer;
    // Tensor mTempDstBuffer;
    Tensor mTempRemainBuffer;
    void (*mGemmKernel)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                        size_t dst_depth_quad, const QuanPostTreatParameters* post);
};

#if defined(__aarch64__) && defined(ENABLE_ARMV82)
class CPUConvArm82Int8 : public CPUConvolution {
public:
    CPUConvArm82Int8(Backend *backend, const MNN::Convolution2D *convParam);
    virtual ~CPUConvArm82Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    // relu or relu6
    bool mRelu;

    std::shared_ptr<Tensor> mWeightInt8;
    std::shared_ptr<Tensor> mBiasInt32;
    std::shared_ptr<Tensor> mScaleFloat;

    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;

    Tensor mTempIm2ColBuffer;
    Tensor mTempRemainBuffer;
};
#endif

} // namespace MNN

#endif /* CPUConvInt8_hpp */
