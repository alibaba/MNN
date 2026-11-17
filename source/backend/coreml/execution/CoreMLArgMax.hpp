//
//  CoreMLArgMax.hpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLARGMAX_HPP
#define MNN_COREMLARGMAX_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLArgMax : public CoreMLCommonExecution {
public:
    CoreMLArgMax(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLArgMax() = default;
};

class CoreMLTopK : public CoreMLCommonExecution {
public:
    CoreMLTopK(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLTopK() = default;
};
} // namespace MNN

#endif // MNN_COREMLARGMAX_HPP
