//
//  ConvolutionPackFreeWinograd.hpp
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#ifndef ConvolutionPackFreeWinograd_hpp
#define ConvolutionPackFreeWinograd_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/ConvolutionWinogradImpl.hpp"
#include "backend/cpu/compute/ConvolutionPackFreeWinograd.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

#define CONVOLUTION_WINOGRAD_MAX_UNIT 8
#define CONVOLUTION_WINOGRAD_MIN_UNIT 2

namespace MNN {

class ConvolutionPackFreeWinograd : public ConvolutionWinogradImpl {
public:
    ConvolutionPackFreeWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output, Backend *b,
                        const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize,
                        WinogradConfig config);
    virtual ~ConvolutionPackFreeWinograd();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    bool updateWinogradBuffer(const Tensor* input, const Tensor* output);
    static WinogradConfig bestWinogradUnit(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                int threadnumber, Backend* b, const PerfConfig& denseConfig);
    static WinogradConfig updateBestWinogradUnit(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                int threadnumber, Backend* b);
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    ConvolutionPackFreeWinograd(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *convOp, Backend* b) : ConvolutionWinogradImpl(convOp, b) {
        mResource = resource;
    }
};
} // namespace MNN
#endif /* ConvolutionWinogradImpl_hpp */
