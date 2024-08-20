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
    enum ConvertType {
        INT8_TO_FlOAT = 0,
        FlOAT_TO_INT8 = 1,
    };
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override;
    static ErrorCode cast(const Tensor* input, const Tensor* output, const CPUBackend* bn, ConvertType type);
    static ErrorCode cast(const void* inputRaw, void* outputRaw, ConvertType type, int number, float scale, float zero, float min, float max, const CPUBackend* bn);
};
} // namespace MNN
#endif /* CPUCast_hpp */
