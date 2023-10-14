//
//  CastExecution.hpp
//  MNN
//
//  Created by MNN on 2023/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CastExecution_hpp
#define CastExecution_hpp

#include "core/Execution.hpp"

#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

class CastExecution : public Execution {
public:
    CastExecution(Backend* bn, DataType dstType) : Execution(bn) {
        mDst = dstType;
    }
    virtual ~CastExecution() = default;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
private:
    DataType mDst;
};

class CastCreator : public CUDABackend::Creator {
public:
    enum ConvertType {
        INT8_TO_FlOAT = 0,
        FlOAT_TO_INT8 = 1,
    };
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override;
    static ErrorCode cast(const Tensor* input, const Tensor* output, Backend* bn, ConvertType type);
    static ErrorCode cast(const Tensor* input, const Tensor* output, ConvertType type, float scale, float zero, float min, float max, Backend* bn);
};

} // namespace CUDA
} // namespace MNN
#endif /* CastExecution_hpp */
