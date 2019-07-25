//
//  CPUSpaceToDepth.hpp
//  MNN
//
//  Created by MNN on 2019/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSpaceToDepth_hpp
#define CPUSpaceToDepth_hpp

#include "Execution.hpp"

namespace MNN {

template <typename T>
class CPUSpaceToDepth : public Execution {
public:
    CPUSpaceToDepth(Backend* backend, const MNN::Op* op);
    virtual ~CPUSpaceToDepth() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;

private:
    const MNN::Op* mOp;
};

} // namespace MNN

#endif /* CPUSpaceToDepth */
