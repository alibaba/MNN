//
//  NNAPICommonExecution.hpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPICOMMONEXECUTION_HPP
#define MNN_NNAPICOMMONEXECUTION_HPP
#include "core/Execution.hpp"
#include "NNAPIBackend.hpp"
#include <memory>

namespace MNN {

class NNAPICommonExecution : public Execution {
public:
    NNAPICommonExecution(Backend *backend, const Op *op);
    virtual ~NNAPICommonExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
protected:
    bool mNCHW;
    NNAPIBackend* mNNAPIBackend;
    const Op* mOp;
    std::vector<uint32_t> getTensorIdxs(const std::vector<Tensor*>& tensors);
    template <typename T> inline uint32_t buildScalar(T scalar) { return mNNAPIBackend->buildScalar(scalar); }
    uint32_t buildConstant(const void* data, size_t size, OperandCode dtype, std::vector<uint32_t> dims = {}, const float* scales = nullptr, int zero = 0);
    uint32_t buildVector(const std::vector<int32_t>& vec);
    uint32_t buildVector(const std::vector<float>& vec);
    uint32_t buildTensor(OperandCode dtype, std::vector<int> dims);
    ErrorCode buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs);
    int formatAxis(int axis, const Tensor* t);
};

} // namespace MNN
#endif // MNN_NNAPICOMMONEXECUTION_HPP
