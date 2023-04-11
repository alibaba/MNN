//
//  NNAPIConvolution.hpp
//  MNN
//
//  Created by MNN on 2022/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPICONVOLUTION_HPP
#define MNN_NNAPICONVOLUTION_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NNAPIConvolution : public NNAPICommonExecution {
public:
    NNAPIConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIConvolution() = default;
private:
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    std::unique_ptr<float[]> nhwcWeight;
    std::unique_ptr<int8_t[]> quantWeight;
    std::unique_ptr<int32_t[]> quantBias;
    bool isDepthwise = false, isDeconv = false;
};
} // namespace MNN

#endif // MNN_NNAPICONVOLUTION_HPP
