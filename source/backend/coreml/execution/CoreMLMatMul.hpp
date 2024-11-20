//
//  CoreMLMatMul.hpp
//  MNN
//
//  Created by MNN on 2024/10/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLMATMUL_HPP
#define MNN_COREMLMATMUL_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLMatMul : public CoreMLCommonExecution {
public:
    CoreMLMatMul(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLMatMul() = default;
};
} // namespace MNN

#endif // MNN_COREMLMATMUL_HPP
