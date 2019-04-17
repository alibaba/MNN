//
//  Convolution3x3.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Convolution3x3_hpp
#define Convolution3x3_hpp

#include "CPUConvolution.hpp"
#include "ConvolutionFloatFactory.h"

namespace MNN {
class Convolution3x3 : public CPUConvolution {
public:
    Convolution3x3(const Convolution2DCommon *convOp, Backend *b, const float *originWeight, size_t originWeightSize,
                   const float *bias, size_t biasSize);
    virtual ~Convolution3x3();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static void sourceTransform(const float *srcBlock, float *dstStart, size_t step);
    static void destTransform(const float *srcZ, float *dstBlock, size_t step);
    static void kernelTransform(float *reorderedWeight, const float *srcWeight, int srcCount, int outputCount);

private:
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;

    Tensor mTempBuffer;
    bool mInsideThread  = false;
    bool mOutsideThread = true;
};
} // namespace MNN
#endif /* Convolution3x3_hpp */
