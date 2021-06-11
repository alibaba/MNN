//
//  CPUGatherV2.hpp
//  MNN
//
//  Created by MNN on 2018/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUGatherV2_hpp
#define CPUGatherV2_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUGatherV2 : public Execution {
public:
    CPUGatherV2(Backend *b, const Op* op);
    virtual ~CPUGatherV2() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    int mAxis;
    const Op* mOp;
};
} // namespace MNN
#endif /* CPUGatherV2_hpp */
