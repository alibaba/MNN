//
//  CPUTopKV2.hpp
//  MNN
//
//  Created by MNN on 2018/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTOPKV2_HPP
#define CPUTOPKV2_HPP

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUTopKV2 : public Execution {
public:
    CPUTopKV2(Backend *b, const Op* op);
    virtual ~CPUTopKV2() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    ErrorCode routeSigmoid(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    bool mLargest = true;
    bool mRouteSigmoid = false;
};
} // namespace MNN

#endif // CPUTOPKV2_HPP
