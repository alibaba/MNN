//
//  ConvolutionWinogradImpl.cpp
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionWinogradImpl.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/WingoradGenerater.hpp"
#include <MNN/AutoTime.hpp>
#include "core/MemoryFormater.h"


//#define MNN_WINOGRAD_PRINT_REDUCE_RATE
//#define MNN_WINO_TRANFORM_TEST_CLOSE
namespace MNN {
ConvolutionWinogradImpl::ConvolutionWinogradImpl(const Convolution2DCommon *convOp, Backend *b)
    : MNN::CPUConvolution(convOp, b) {

}

ConvolutionWinogradImpl::~ConvolutionWinogradImpl() {

}


WinogradConfig ConvolutionWinogradImpl::bestWinogradUnit(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b, const PerfConfig& denseConfig) {
    return WinogradConfig();
}

bool ConvolutionWinogradImpl::canUseWinograd(const Convolution2DCommon *common) {
    if (common->kernelY() != common->kernelX() || common->kernelY() <= 1) {
        return false;
    }
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    return true;
}

} // namespace MNN
