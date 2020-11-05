//
//  TRTDepthwiseConvolution.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTDepthwiseConvolution_HPP
#define MNN_TRTDepthwiseConvolution_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTDepthwiseConvolution : public TRTCommonExecution {
public:
    TRTDepthwiseConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs);
    virtual ~TRTDepthwiseConvolution() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;

private:
    IActivationLayer *mActivationLayer{nullptr};
};

} // namespace MNN

#endif // MNN_TRTDepthwiseConvolution_HPP
