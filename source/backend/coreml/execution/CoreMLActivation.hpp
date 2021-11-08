//
//  CoreMLActivation.hpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLACTIVATION_HPP
#define MNN_COREMLACTIVATION_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLActivation : public CoreMLCommonExecution {
public:
    CoreMLActivation(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLActivation() = default;
};
} // namespace MNN

#endif // MNN_COREMLACTIVATION_HPP
