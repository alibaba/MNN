//
//  ConvolutionIntFactory.cpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionIntFactory.hpp"
#include "backend/cpu/compute/ConvolutionGroup.hpp"
#include "backend/cpu/compute/IdstConvolutionInt8.hpp"

namespace MNN {
Execution *ConvolutionIntFactory::createUnit(const Tensor *input, const Tensor *output, const MNN::Op *op,
                                             Backend *backend, const ConvolutionCommon::Int8Common *common, const float *bias,
                                             size_t biasSize) {
    auto conv2d = op->main_as_Convolution2D();
    return new IdstConvolutionInt8(conv2d->common(), backend, common, bias, biasSize);
}

Execution *ConvolutionIntFactory::create(const Tensor *input, const Tensor *output, const MNN::Op *op, Backend *backend,
                                         const ConvolutionCommon::Int8Common *common) {
    auto conv2d = op->main_as_Convolution2D();
    int group            = conv2d->common()->group();
    if (conv2d->common()->inputCount() != input->channel() && conv2d->common()->inputCount() > 0) {
        group = input->channel()/ conv2d->common()->inputCount();
    }
    if (1 == group) {
        return createUnit(input, output, op, backend, common, conv2d->bias()->data(), conv2d->bias()->size());
    }
    MNN_ASSERT(common->weight.get() != nullptr);

    // Split
    std::vector<std::shared_ptr<Execution>> subConvolution;
    auto groupOutputCount = conv2d->common()->outputCount() / group;
    auto groupWeightSize  = common->weight.size() / group;
    for (int i = 0; i < group; ++i) {
        auto subCommon = std::make_shared<ConvolutionCommon::Int8Common>();
        subCommon->alpha.reset(groupOutputCount);
        ::memcpy(subCommon->alpha.get(), common->alpha.get() + groupOutputCount * i, groupOutputCount * sizeof(float));
        subCommon->quan = common->quan;
        subCommon->weight.reset(groupWeightSize);
        ::memcpy(subCommon->weight.get(), common->weight.get() + groupWeightSize * i, groupWeightSize * sizeof(int8_t));
        subConvolution.push_back(
            std::shared_ptr<Execution>(createUnit(input, output, op, backend, subCommon.get(),
                                                  conv2d->bias()->data() + groupOutputCount * i, groupOutputCount)));
    }
    return new ConvolutionGroup(backend, subConvolution);
}

} // namespace MNN
