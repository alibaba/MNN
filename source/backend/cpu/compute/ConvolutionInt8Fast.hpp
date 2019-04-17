//
//  ConvolutionInt8Fast.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionInt8Fast_hpp
#define ConvolutionInt8Fast_hpp

#include <stdio.h>
#include "AutoStorage.h"
#include "CPUConvolution.hpp"
#include "ConvolutionFloatFactory.h"
#include "ConvolutionIntFactory.hpp"
#include "Int8FunctionsOpt.h"

namespace MNN {
class ConvolutionInt8Fast : public CPUConvolution {
public:
    ConvolutionInt8Fast(const Convolution2DCommon* convOp, Backend* b, const ConvolutionIntFactory::Int8Common* common,
                        const float* bias, size_t biasSize);
    virtual ~ConvolutionInt8Fast() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    AutoStorage<int8_t> mWeight;
    AutoStorage<float> mAlpha;
    AutoStorage<float> mBias;
    const IDSTQuan* mQuan;
    int mSrcCount;
};
} // namespace MNN

#endif /* ConvolutionInt8Fast_hpp */
