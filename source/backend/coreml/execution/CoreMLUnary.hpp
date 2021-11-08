//
//  CoreMLUnary.hpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLUNARY_HPP
#define MNN_COREMLUNARY_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLUnary : public CoreMLCommonExecution {
public:
    CoreMLUnary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLUnary() = default;
};
} // namespace MNN

#endif // MNN_COREMLUNARY_HPP
