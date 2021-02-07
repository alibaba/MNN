//
//  NPUPooling.hpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUPOOLING_HPP
#define MNN_NPUPOOLING_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUPooling : public NPUCommonExecution {
public:
    NPUPooling(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUPooling() = default;
};

} // namespace MNN

#endif // MNN_NPUPOOLING_HPP
