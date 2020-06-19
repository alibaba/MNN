//
//  Arm82Padding.hpp
//  MNN
//
//  Created by MNN on 2020/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#ifndef Arm82Padding_hpp
#define Arm82Padding_hpp

#include "backend/arm82/Arm82Backend.hpp"
namespace MNN {

class Arm82PaddingPacked : public Execution {
public:
    Arm82PaddingPacked(Backend *bn, PadValueMode mode) : Execution(bn), mMode(mode) {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempInput;
    std::shared_ptr<Tensor> mTempOutput;
    std::vector<Tensor *> mTempInputs;
    std::vector<Tensor *> mTempOutputs;
    bool mNeedConvert = false;
    PadValueMode mMode;
    Tensor mCache;
};

class Arm82Padding : public Execution {
public:
    Arm82Padding(Backend *bn, PadValueMode mode) : Execution(bn), mMode(mode) {
        // Do nothing
    }
    static void execute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                        PadValueMode mode = PadValueMode_CONSTANT);
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Tensor mCache;
    PadValueMode mMode;
};
}; // namespace MNN

#endif

#endif
