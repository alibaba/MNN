//
//  CPUSelect.hpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef CPUSelect_hpp
#define CPUSelect_hpp

#include "backend/cpu/CPUBackend.hpp"
namespace MNN {
class CPUSelect : public Execution {
public:
    CPUSelect(Backend *bn) : Execution(bn) {
        // Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
}; // namespace MNN

#endif /* CPUSelect_hpp */
