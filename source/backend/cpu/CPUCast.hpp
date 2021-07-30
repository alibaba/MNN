//
//  CPUCast.hpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUCast_hpp
#define CPUCast_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
class CPUCastCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override;
    static ErrorCode cast(const Tensor* input, const Tensor* output, int size = 0);
    static ErrorCode cast(void* const inputRaw, void* outputRaw, halide_type_t inputType, halide_type_t outputType, int number, float scale, float zero, float min, float max);
};
} // namespace MNN
#endif /* CPUCast_hpp */
