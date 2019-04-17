//
//  CPUBatchMatMul.hpp
//  MNN
//
//  Created by MNN on 2019/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBatchMatMul_hpp
#define CPUBatchMatMul_hpp

#include "Execution.hpp"

namespace MNN {

class CPUBatchMatMul : public Execution {
public:
    CPUBatchMatMul(const Op *op, Backend *backend);
    virtual ~CPUBatchMatMul() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mBatch;
    std::shared_ptr<Tensor> mMatrixA;
    std::shared_ptr<Tensor> mMatrixB;
    std::shared_ptr<Tensor> mMatrixC;
};

} // namespace MNN

#endif /* CPUBatchMatMul_hpp */
