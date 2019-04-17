//
//  CPUBatchToSpaceND.hpp
//  MNN
//
//  Created by MNN on 2018/12/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBatchToSpaceND_hpp
#define CPUBatchToSpaceND_hpp

#include <functional>
#include "Execution.hpp"
namespace MNN {

class CPUBatchToSpaceND : public Execution {
public:
    CPUBatchToSpaceND(const Op *op, Backend *bn);
    virtual ~CPUBatchToSpaceND() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Op *mOp;
    std::function<void()> mRun;
};

} // namespace MNN

#endif /* CPUBatchToSpaceND_hpp */
