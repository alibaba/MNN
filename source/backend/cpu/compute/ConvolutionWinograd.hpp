//
//  ConvolutionWinograd.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionWinograd_hpp
#define ConvolutionWinograd_hpp

#include "CPUConvolution.hpp"
#include "ConvolutionFloatFactory.h"
#include "WinogradOptFunction.hpp"

namespace MNN {
class ConvolutionWinograd : public CPUConvolution {
public:
    ConvolutionWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output, Backend *b,
                        const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize,
                        int unit);
    virtual ~ConvolutionWinograd();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static bool canUseWinograd(const Convolution2DCommon *convOp);
    static int bestWinogradUnit(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                int threadnumber);

private:
    std::shared_ptr<Tensor> mBias;
    std::shared_ptr<Tensor> mA;
    std::shared_ptr<Tensor> mB;
    std::shared_ptr<Tensor> mWeight;

    Tensor mTempBuffer;
    Tensor mTransformMidBuffer;

    WinogradFunction::TransformFunc mSourceTransform;
    WinogradFunction::TransformFunc mDestTransform;
};
} // namespace MNN
#endif /* ConvolutionWinograd_hpp */
