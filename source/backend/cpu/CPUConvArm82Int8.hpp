//
//  CPUConvArm82Int8.hpp
//  MNN
//
//  Created by MNN on b'2020/12/30'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef CPUConvArm82Int8_hpp
#define CPUConvArm82Int8_hpp
#if defined(ENABLE_ARMV82) && (defined(__ANDROID__) || defined(__aarch64__))
#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/CPUConvInt8.hpp"
#include <MNN/Tensor.hpp>
namespace MNN {
class CPUConvArm82Int8 : public CPUConvolution {
public:
    CPUConvArm82Int8(Backend *backend, const MNN::Convolution2D *convParam, float inputScale, float outputScale);
    CPUConvArm82Int8(std::shared_ptr<CPUConvInt8::ResourceInt8> res, Backend* backend, const MNN::Convolution2DCommon* common);

    virtual ~CPUConvArm82Int8() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    // relu or relu6
    bool mRelu;
    std::shared_ptr<CPUConvInt8::ResourceInt8> mResource;
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;

    Tensor mTempIm2ColBuffer;
    Tensor mTempRemainBuffer;
};
};
#endif
#endif
