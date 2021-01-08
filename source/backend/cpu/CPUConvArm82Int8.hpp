//
//  CPUConvArm82Int8.hpp
//  MNN
//
//  Created by MNN on b'2020/12/30'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef CPUConvArm82Int8_hpp
#define CPUConvArm82Int8_hpp
#if defined(__aarch64__) && defined(ENABLE_ARMV82)
#include "backend/cpu/CPUConvolution.hpp"
#include <MNN/Tensor.hpp>
namespace MNN {
class CPUConvArm82Int8 : public CPUConvolution {
public:
    struct Resource {
        std::shared_ptr<Tensor> mWeightInt8;
        std::shared_ptr<Tensor> mBiasInt32;
        std::shared_ptr<Tensor> mScaleFloat;
        Backend* backend;
        ~ Resource();
    };
    CPUConvArm82Int8(Backend *backend, const MNN::Convolution2D *convParam);
    CPUConvArm82Int8(std::shared_ptr<CPUConvArm82Int8::Resource> res, Backend* backend, const MNN::Convolution2DCommon* common);

    virtual ~CPUConvArm82Int8() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    // relu or relu6
    bool mRelu;
    std::shared_ptr<CPUConvArm82Int8::Resource> mResource;


    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;

    Tensor mTempIm2ColBuffer;
    Tensor mTempRemainBuffer;
};
};
#endif
#endif
