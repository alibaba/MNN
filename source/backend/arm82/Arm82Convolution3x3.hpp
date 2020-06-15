//
//  Arm82Convolution3x3.hpp
//  MNN
//
//  Created by MNN on 2020/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#ifndef Arm82Convolution3x3_hpp
#define Arm82Convolution3x3_hpp

#include "backend/arm82/Arm82Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {
class Arm82Convolution3x3 : public Execution {
public:
    Arm82Convolution3x3(const MNN::Convolution2D *convParam, Backend *bn);
    virtual ~Arm82Convolution3x3();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mTileCount;
    int mThreadNums;
    int mPadX;
    int mPadY;
    bool mRelu;
    bool mRelu6;
    std::shared_ptr<Tensor> mWeightFp16;
    std::shared_ptr<Tensor> mBiasFp16;

    Tensor mTransformBuffer;
    Tensor mDummyBias;
    const Convolution2DCommon *mCommon;
};

} // namespace MNN

#endif

#endif
