//
//  CoreMLInterp.hpp
//  MNN
//
//  Created by MNN on 2021/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLINTERP_HPP
#define MNN_COREMLINTERP_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLInterp : public CoreMLCommonExecution {
public:
    CoreMLInterp(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLInterp() = default;
};
} // namespace MNN

#endif // MNN_COREMLINTERP_HPP
