//
//  ConvolutionPackWinograd.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionPackWinograd_hpp
#define ConvolutionPackWinograd_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/ConvolutionWinogradImpl.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

#define CONVOLUTION_WINOGRAD_MAX_UNIT 8
#define CONVOLUTION_WINOGRAD_MIN_UNIT 2

namespace MNN {
class ConvolutionPackWinograd : public ConvolutionWinogradImpl {
public:
    ConvolutionPackWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output, Backend *b,
                        const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize,
                        WinogradConfig config);
    virtual ~ConvolutionPackWinograd();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static WinogradConfig bestWinogradUnit(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                int threadnumber, Backend* b, const PerfConfig& denseConfig);
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    ConvolutionPackWinograd(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *convOp, Backend* b)
    : ConvolutionWinogradImpl(convOp, b) {
        mResource = resource;
    }
    std::pair<int, std::function<void(int tId, const uint8_t*, uint8_t*)>> mMainFunction;
    std::pair<int, std::function<void(int, uint8_t*)>> mPostFunction;

};
} // namespace MNN
#endif /* ConvolutionPackWinograd_hpp */
