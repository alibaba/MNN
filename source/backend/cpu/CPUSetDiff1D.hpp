//
//  CPUSetDiff1D.hpp
//  MNN
//
//  Created by MNN on 2019/6/11.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef CPUSetDiff1D_hpp
#define CPUSetDiff1D_hpp

#include "backend/cpu/CPUBackend.hpp"
namespace MNN {
class CPUSetDiff1D : public Execution {
public:
    CPUSetDiff1D(Backend *bn) : Execution(bn) {
        // Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
};     // namespace MNN
#endif /* CPUSetDiff1D_hpp */
