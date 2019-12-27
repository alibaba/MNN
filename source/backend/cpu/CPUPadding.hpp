//
//  CPUPadding.hpp
//  MNN
//
//  Created by MNN on 2019/6/24.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef CPUPadding_hpp
#define CPUPadding_hpp

#include <stdio.h>
#include "backend/cpu/CPUBackend.hpp"
namespace MNN {
class CPUPaddingPacked : public Execution {
public:
    CPUPaddingPacked(Backend *bn, PadValueMode mode) : Execution(bn), mMode(mode) {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    std::shared_ptr<Tensor> mTempInput;
    std::shared_ptr<Tensor> mTempOutput;
    std::vector<Tensor*> mTempInputs;
    std::vector<Tensor*> mTempOutputs;
    bool mNeedConvert = false;
    PadValueMode mMode;
    Tensor mCache;
};
class CPUPadding : public Execution {
public:
    CPUPadding(Backend *bn, PadValueMode mode) : Execution(bn), mMode(mode) {
        // Do nothing
    }
    static void execute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, PadValueMode mode = PadValueMode_CONSTANT);
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
private:
    Tensor mCache;
    PadValueMode mMode;
};
}; // namespace MNN

#endif /* CPUPadding_hpp */
