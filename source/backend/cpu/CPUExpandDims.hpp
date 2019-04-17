//
//  CPUExpandDims.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUExpandDims_hpp
#define CPUExpandDims_hpp

#include "Execution.hpp"

namespace MNN {
class CPUExpandDims : public Execution {
public:
    CPUExpandDims(Backend *b);
    virtual ~CPUExpandDims() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace MNN
#endif /* CPUExpandDims_hpp */
