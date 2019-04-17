//
//  CPUConst.hpp
//  MNN
//
//  Created by MNN on 2018/08/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConst_hpp
#define CPUConst_hpp

#include "Execution.hpp"

namespace MNN {
class CPUConst : public Execution {
public:
    CPUConst(Backend *b, const MNN::Op *op);
    virtual ~CPUConst() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    const MNN::Op *mOp;
};
} // namespace MNN
#endif /* CPUConst_hpp */
