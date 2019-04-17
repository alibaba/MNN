//
//  CPUMatMul.hpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUMATMUL_HPP
#define CPUMATMUL_HPP

#include "Execution.hpp"

namespace MNN {

class CPUMatMul : public Execution {
public:
    CPUMatMul(Backend *backend, bool transposeA, bool transposeB);
    virtual ~CPUMatMul() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool mTransposeA;
    bool mTransposeB;
};
} // namespace MNN

#endif // CPUMATMUL_HPP
