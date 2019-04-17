//
//  CPUReduction.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUReduction_hpp
#define CPUReduction_hpp

#include "CPUBackend.hpp"

namespace MNN {
class CPUReductionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override;
};
} // namespace MNN
#endif /* CPUReduction_hpp */
