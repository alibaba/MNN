//
//  NeuronAdapterConvolution.hpp
//  MNN
//
//  Created by MNN on 2022/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NeuronAdapterCONVOLUTION_HPP
#define MNN_NeuronAdapterCONVOLUTION_HPP

#include "NeuronAdapterBackend.hpp"
#include "NeuronAdapterCommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NeuronAdapterConvolution : public NeuronAdapterCommonExecution {
public:
    NeuronAdapterConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NeuronAdapterConvolution() = default;
private:
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    std::unique_ptr<float[]> nhwcWeight;
    std::unique_ptr<int8_t[]> quantWeight;
    std::unique_ptr<int32_t[]> quantBias;
    bool isDepthwise = false, isDeconv = false;
};
} // namespace MNN

#endif // MNN_NeuronAdapterCONVOLUTION_HPP
