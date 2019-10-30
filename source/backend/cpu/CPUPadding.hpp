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
#include "CPUBackend.hpp"
namespace MNN {
class CPUPaddingPacked : public Execution {
public:
    CPUPaddingPacked(Backend *bn) : Execution(bn) {
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
};
class CPUPadding : public Execution {
public:
    CPUPadding(Backend *bn) : Execution(bn) {
        // Do nothing
    }
    static void execute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        execute(inputs, outputs);
        return NO_ERROR;
    }
};
}; // namespace MNN

#endif /* CPUPadding_hpp */
