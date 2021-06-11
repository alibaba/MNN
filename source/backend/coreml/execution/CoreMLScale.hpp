//
//  CoreMLScale.hpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLSCALE_HPP
#define MNN_COREMLSCALE_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLScale : public CoreMLCommonExecution {
public:
    CoreMLScale(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLScale() = default;
};
} // namespace MNN

#endif // MNN_COREMLSCALE_HPP
