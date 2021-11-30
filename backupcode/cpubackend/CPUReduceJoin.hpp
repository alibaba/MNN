//
//  CPUReduceJoin.hpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUReduceJoin_hpp
#define CPUReduceJoin_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
class CPUReduceJoinCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override;
};
} // namespace MNN
#endif /* CPUReduceJoin_hpp */
