//
//  ConvolutionGroup.hpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionGroupInt8_hpp
#define ConvolutionGroupInt8_hpp

#include "backend/cpu/compute/ConvolutionIntFactory.hpp"

namespace MNN {
class ConvolutionGroup : public Execution {
public:
    ConvolutionGroup(Backend *b, const std::vector<std::shared_ptr<Execution>> &subConvolution);
    virtual ~ConvolutionGroup() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::unique_ptr<Tensor> mInputRaw;
    std::unique_ptr<Tensor> mOutputRaw;

    std::unique_ptr<Tensor> mInputUnit;
    std::unique_ptr<Tensor> mOutputUnit;

    std::vector<Tensor *> mInputUnitWrap;
    std::vector<Tensor *> mOutputUnitWrap;
    std::vector<std::shared_ptr<Execution>> mSubConvolution;
};
} // namespace MNN

#endif /* ConvolutionGroupInt8_hpp */
