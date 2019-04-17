//
//  ConvolutionDepthwise3x3.hpp
//  MNN
//
//  Created by MNN on 2019/4/3.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionDepthwise3x3_hpp
#define ConvolutionDepthwise3x3_hpp

#include "CPUConvolution.hpp"

namespace MNN {
class ConvolutionDepthwise3x3 : public CPUConvolution {
public:
    ConvolutionDepthwise3x3(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                            size_t originWeightSize, const float *bias, size_t biasSize);
    virtual ~ConvolutionDepthwise3x3();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::unique_ptr<Tensor> mWeight;
    std::unique_ptr<Tensor> mBias;

    std::unique_ptr<Tensor> mCacheLine;
    int mSourceStartX = 0;
    int mSourceEndX   = 0;
};
} // namespace MNN

#endif /* ConvolutionDepthwise3x3_hpp */
