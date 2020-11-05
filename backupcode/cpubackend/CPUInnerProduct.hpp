//
//  CPUInnerProduct.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInnerProduct_hpp
#define CPUInnerProduct_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
class CPUInnerProductCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override;
};
} // namespace MNN
#endif /* CPUMatMul_hpp */
