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
    CoreMLConvolution(Backend* b, const Op* op, const std::vector<Tensor*>& inputs,
                      const std::vector<Tensor*>& outputs);
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    virtual ~CoreMLConvolution() = default;

private:
    void addOutput(Operation, const std::vector<Tensor*>&);
    void initConv2d(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    bool mIsDeconv = false;
    bool mIsLinear = false;
    std::string mWeight, mBias;
};

class CoreMLConvolution3D : public CoreMLCommonExecution {
public:
    CoreMLConvolution3D(Backend* b, const Op* op, const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs);
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    virtual ~CoreMLConvolution3D() = default;

private:
    std::string mWeight, mBias;
};

} // namespace MNN

#endif // MNN_COREMLCONVOLUTION_HPP
