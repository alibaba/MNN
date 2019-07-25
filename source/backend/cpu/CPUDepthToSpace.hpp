//
//  CPUDepthToSpace.hpp
//  MNN
//
//  Created by MNN on 2019/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDepthToSpace_hpp
#define CPUDepthToSpace_hpp

#include "Execution.hpp"

namespace MNN {

template <typename T>
class CPUDepthToSpace : public Execution {
public:
    CPUDepthToSpace(Backend* backend, const MNN::Op* op);
    virtual ~CPUDepthToSpace() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;

private:
    const MNN::Op* mOp;
};

} // namespace MNN

#endif /* CPUDepthToSpace */
