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
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
class CPUPadding : public Execution {
public:
    CPUPadding(Backend *bn) : Execution(bn) {
        // Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
}; // namespace MNN

#endif /* CPUPadding_hpp */
