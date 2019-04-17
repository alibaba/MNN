//
//  CPUCast.hpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUCast_hpp
#define CPUCast_hpp

#include "CPUBackend.hpp"

namespace MNN {
class CPUCastCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override;
};
} // namespace MNN
#endif /* CPUCast_hpp */
