//
//  CoreMLRaster.hpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLRASTER_HPP
#define MNN_COREMLRASTER_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLRaster : public CoreMLCommonExecution {
public:
    CoreMLRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLRaster() = default;
};
} // namespace MNN

#endif // MNN_COREMLRASTER_HPP
