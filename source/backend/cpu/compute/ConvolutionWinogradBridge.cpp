//
//  ConvolutionWinogradBridge.cpp
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/ConvolutionWinogradImpl.hpp"
#include "backend/cpu/compute/ConvolutionWinogradBridge.hpp"
#include "backend/cpu/compute/ConvolutionPackFreeWinograd.hpp"
#include "backend/cpu/compute/ConvolutionPackWinograd.hpp"

namespace MNN {


WinogradConfig ConvolutionWinogradBridge::bestWinogradUnit(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b, const PerfConfig& denseConfig) {

//  Currently packfree is only used in x86 architecture
#ifdef MNN_USE_SSE
    auto core = static_cast<CPUBackend*>(b)->functions();
    if (16 == core->pack) { // avx512
        return ConvolutionPackFreeWinograd::bestWinogradUnit(common, inputTensor, outputTensor, threadNumber, b, denseConfig);
    } else {
#endif
        return ConvolutionPackWinograd::bestWinogradUnit(common, inputTensor, outputTensor, threadNumber, b, denseConfig);

#ifdef MNN_USE_SSE
    }
#endif

}

bool ConvolutionWinogradBridge::canUseWinograd(const Convolution2DCommon *common) {
    return ConvolutionPackWinograd::canUseWinograd(common);
}

ConvolutionWinogradImpl *ConvolutionWinogradBridge::createWinogradImpl(const Convolution2DCommon *common,
                                                                       const Tensor *input, const Tensor *output,
                                                                       Backend *b, const float *originWeight,
                                                                       size_t originWeightSize, const float *bias,
                                                                       size_t biasSize, WinogradConfig config) {

#ifdef MNN_USE_SSE
    auto core = static_cast<CPUBackend*>(b)->functions();
    // Adopt different algorithm for x86 and arm
    if (16 == core->pack) { // avx512
        return new ConvolutionPackFreeWinograd(common, input, output, b, originWeight, originWeightSize, bias, biasSize,
                                   config);
    } else {
#endif

        return new ConvolutionPackWinograd(common, input, output, b, originWeight, originWeightSize, bias, biasSize,
                                   config);
#ifdef MNN_USE_SSE
    }
#endif
}

} // namespace MNN
