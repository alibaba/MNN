//
//  ConvolutionTiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionTiledExecutor_hpp
#define ConvolutionTiledExecutor_hpp

#include <functional>
#include "../CPUConvolution.hpp"
#include "AutoStorage.h"

// Tiled Slide Window Algorithm
namespace MNN {
class ConvolutionTiledExecutor : public CPUConvolution {
public:
    ConvolutionTiledExecutor(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                             size_t originWeightSize, const float *bias, size_t biasSize);
    virtual ~ConvolutionTiledExecutor();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;
    int mSrcCount;

    Tensor mTempBuffer;
    std::vector<std::pair<int, std::function<void(int)>>> mFunctions;
};
} // namespace MNN

#endif /* ConvolutionTiledExecutor_hpp */
