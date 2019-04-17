//
//  CPUGather.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUGather_hpp
#define CPUGather_hpp

#include "Execution.hpp"

namespace MNN {
class CPUGather : public Execution {
public:
    CPUGather(Backend *b, const MNN::Op *op);
    virtual ~CPUGather() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    const MNN::Op *mOp;
};
} // namespace MNN
#endif /* CPUGather_hpp */
