//
//  VulkanConvolutionWrap.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanConvolutionWrap.hpp"
#include "ConvolutionIntFactory.hpp"
#include "Macro.h"
#include "VulkanGroupConvolution.hpp"
namespace MNN {
VulkanConvolutionWrap::VulkanConvolutionWrap(const Op *op, Backend *backend) : Execution(backend) {
    mConvParameter = op->main_as_Convolution2D();
}
VulkanConvolutionWrap::~VulkanConvolutionWrap() {
}

ErrorCode VulkanConvolutionWrap::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return mEncodeConvolution->onExecute(inputs, outputs);
}
ErrorCode VulkanConvolutionWrap::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    if (nullptr == mEncodeConvolution) {
        auto extra          = static_cast<VulkanBackend *>(backend());
        auto convReal       = mConvParameter;
        auto common         = convReal->common();
        auto outputCount    = common->outputCount();
        const int fh        = common->kernelY();
        const int fw        = common->kernelX();
        int srcCount        = 0;
        const float *source = nullptr;
        std::shared_ptr<ConvolutionIntFactory::Int8Common> quanCommon;
        // check whether idst quantized op
        if (nullptr != convReal->quanParameter()) {
            quanCommon = ConvolutionIntFactory::load(convReal->quanParameter(), true);
            srcCount   = quanCommon->weightFloat.size() / (outputCount * fh * fw);
            source     = quanCommon->weightFloat.get();
        } else {
            srcCount = convReal->weight()->size() / (outputCount * fh * fw);
            source   = convReal->weight()->data();
        }

        mEncodeConvolution = VulkanConvolutionImpl::create(extra, common, input, output, source,
                                                           convReal->bias()->data(), srcCount, outputCount);
    }
    return mEncodeConvolution->onResize(inputs, outputs);
}

} // namespace MNN
