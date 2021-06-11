//
//  CoreMLReduction.hpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLREDUCTION_HPP
#define MNN_COREMLREDUCTION_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLReduction : public CoreMLCommonExecution {
public:
    CoreMLReduction(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLReduction() = default;
};
} // namespace MNN

#endif // MNN_COREMLREDUCTION_HPP
