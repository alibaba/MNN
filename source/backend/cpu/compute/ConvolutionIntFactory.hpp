//
//  ConvolutionIntFactory.hpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionIntFactory_hpp
#define ConvolutionIntFactory_hpp

#include <stdint.h>
#include <memory>
#include "backend/cpu/CPUBackend.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
class ConvolutionIntFactory {
public:
    static Execution* create(const Tensor* input, const Tensor* output, const MNN::Op* op, Backend* backend,
                             const ConvolutionCommon::Int8Common* common);

    static Execution* createUnit(const Tensor* input, const Tensor* output, const MNN::Op* op, Backend* bn,
                                 const ConvolutionCommon::Int8Common* common, const float* bias, size_t biasSize);
};
} // namespace MNN

#endif /* ConvolutionIntFactory_hpp */
