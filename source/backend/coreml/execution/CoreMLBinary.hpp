//
//  CoreMLBinary.hpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLBIANRY_HPP
#define MNN_COREMLBIANRY_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLBinary : public CoreMLCommonExecution {
public:
    CoreMLBinary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLBinary() = default;
};
} // namespace MNN

#endif // MNN_COREMLBINARY_HPP
