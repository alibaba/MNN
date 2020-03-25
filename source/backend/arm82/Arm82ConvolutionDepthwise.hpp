//
//  Arm82ConvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on 2020/01/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Arm82ConvolutionDepthwise_hpp
#define Arm82ConvolutionDepthwise_hpp

#include "MNN_generated.h"
#include "backend/arm82/Arm82Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {
class Arm82ConvolutionDepthwise : public Execution {
public:
    Arm82ConvolutionDepthwise(const MNN::Convolution2D *convParam, Backend *bn);
    virtual ~Arm82ConvolutionDepthwise();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mWeightFp16;
    std::shared_ptr<Tensor> mBiasFp16;
    const Convolution2DCommon *mCommon;
    int mThreadNumber;
    bool mRelu;
    bool mRelu6;
    std::function<void(int tId, const FLOAT16 *src, FLOAT16 *dst)> mThreadFunction;
};

} // namespace MNN

#endif /* Arm82ConvolutionDepthwise_hpp */
