//
//  Arm82Convolution.hpp
//  MNN
//
//  Created by MNN on 2020/01/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#ifndef Arm82Convolution_hpp
#define Arm82Convolution_hpp

#include "core/ConvolutionCommon.hpp"
#include "core/Execution.hpp"

namespace MNN {
class Arm82Convolution : public Execution {
public:
    Arm82Convolution(const MNN::Convolution2D *convParam, Backend *bn);
    virtual ~Arm82Convolution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    // plane tile number
    int mTileCount;
    int mThreadNums;
    bool mRelu;
    bool mRelu6;
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    std::shared_ptr<Tensor> mWeightFp16;
    std::shared_ptr<Tensor> mBiasFp16;

    Tensor mIm2ColBuffer;
    Tensor mRemainBuffer;
    const Convolution2DCommon *mCommon;
};
} // namespace MNN

#endif /* Arm82Convolution_hpp */
#endif
