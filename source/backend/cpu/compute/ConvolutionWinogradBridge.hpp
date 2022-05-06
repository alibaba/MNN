//
//  ConvolutionWinogradBridge
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#ifndef ConvolutionWinogradBridge_hpp
#define ConvolutionWinogradBridge_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/ConvolutionWinogradImpl.hpp"


namespace MNN {

class ConvolutionWinogradBridge  {
public:
    static bool canUseWinograd(const Convolution2DCommon *convOp);

    static WinogradConfig bestWinogradUnit(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
        int threadnumber, Backend* b, const PerfConfig& denseConfig);

    static ConvolutionWinogradImpl* createWinogradImpl(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output, Backend *b,
        const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize,
        WinogradConfig config);
};

} // namespace MNN
#endif /* ConvolutionWinogradBridge_hpp */
