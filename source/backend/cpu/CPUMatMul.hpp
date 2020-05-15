//
//  CPUMatMul.hpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUMATMUL_HPP
#define CPUMATMUL_HPP

#include <functional>
#include "core/Execution.hpp"
#include "backend/cpu/compute/StrassenMatmulComputor.hpp"

namespace MNN {

class CPUMatMul : public Execution {
public:
    CPUMatMul(Backend *backend, bool transposeA, bool transposeB, bool multiThread);
    virtual ~CPUMatMul() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool mTransposeA;
    bool mTransposeB;
    bool mSupportMultiThread = false;
    std::shared_ptr<Tensor> mAPack;
    std::shared_ptr<Tensor> mBPack;
    std::shared_ptr<Tensor> mCPack;
    const float* mAPtr;
    const float* mBPtr;
    float* mCPtr;
    int mE;
    int mL;
    int mH;
};
} // namespace MNN

#endif // CPUMATMUL_HPP
