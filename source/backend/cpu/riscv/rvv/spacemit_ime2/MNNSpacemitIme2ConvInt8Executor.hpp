//
//  MNNSpacemitIme2ConvInt8Executor.hpp
//  MNN
//
//  Created by MNN on 2026/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_SPACEMIT_IME2_CONV_INT8_EXECUTOR_HPP
#define MNN_SPACEMIT_IME2_CONV_INT8_EXECUTOR_HPP

#include "backend/cpu/compute/ConvInt8TiledExecutor.hpp"

namespace MNN {

class SpacemitIme2ConvInt8Executor : public DenseConvInt8TiledExecutor {
public:
    SpacemitIme2ConvInt8Executor(Backend* backend, const Op* op,
                                 std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant);
    virtual ~SpacemitIme2ConvInt8Executor();
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

protected:
    virtual bool onSetupLinearFastPath(const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs) override;
    virtual DenseConvInt8TiledExecutor* createClone(Backend* bn, const Op* op) const override;

private:
    class LinearResource;
    SpacemitIme2ConvInt8Executor(Backend* backend, const Op* op,
                                 std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant,
                                 std::shared_ptr<LinearResource> linearResource);
    SpacemitIme2ConvInt8Executor(Backend* backend, const Op* op, const SpacemitIme2ConvInt8Executor& exe);
    bool tryExecuteFast(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    std::shared_ptr<LinearResource> mLinearResource;
    bool mUsePrefill = false;
    bool mLinear1x1 = false;
    bool mDecodeBiasChecked = false;
    bool mDecodeBiasAllZero = false;
};

Execution* MNNSpacemitIme2CreateInt8GemmExecution(Backend* backend, const Op* op,
                                                  std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon,
                                                  bool isDynamicQuant);

} // namespace MNN

#endif // MNN_SPACEMIT_IME2_CONV_INT8_EXECUTOR_HPP
