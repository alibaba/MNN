//
//  CPUPermute.hpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUPermute_hpp
#define CPUPermute_hpp

#include "Execution.hpp"

namespace MNN {
class CPUPermute : public Execution {
public:
    CPUPermute(Backend *b, const MNN::Op *op);
    virtual ~CPUPermute() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<int> mDims;
};
} // namespace MNN
#endif /* CPUPermute_hpp */
