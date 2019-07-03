//
//  CPUPool.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUPool_hpp
#define CPUPool_hpp

#include "CPUBackend.hpp"

namespace MNN {
class CPUPool : public Execution {
public:
    CPUPool(Backend *b, const Pool *parameter);
    virtual ~CPUPool() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Pool *mParameter;
    std::pair<int, std::function<void(int)> > mFunction;
};
} // namespace MNN

#endif /* CPUPool_hpp */
