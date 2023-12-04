//
//  CoreMLConvolution.hpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLCONVOLUTION_HPP
#define MNN_COREMLCONVOLUTION_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class CoreMLConvolution : public CoreMLCommonExecution {
public:
    CoreMLConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLConvolution() = default;
private:
    void loadWeightBias(const std::vector<Tensor *> &inputs);
    void addPadLayer(const Tensor * input, const Convolution2DCommon* common);
    std::string mConvInputName, mConvOutputName;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    const float *weightPtr, *biasPtr;
    int weightSize, biasSize;
    bool isDeconv = false;
    bool isSamePadding = false;
    int outputHeight, outputWidth, inputHeight, inputWidth;
};
} // namespace MNN

#endif // MNN_COREMLCONVOLUTION_HPP
