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
#include "../CPUBackend.hpp"
#include "AutoStorage.h"
#include "MNN_generated.h"

namespace MNN {
class ConvolutionIntFactory {
public:
    struct Int8Common {
        AutoStorage<int8_t> weight;
        AutoStorage<float> alpha;
        AutoStorage<float> weightFloat;
        const IDSTQuan* quan;
    };
    MNN_PUBLIC static std::shared_ptr<Int8Common> load(const IDSTQuan* quan, bool forceFloat = false);
    static Execution* create(const Tensor* input, const Tensor* output, const MNN::Op* op, Backend* backend,
                             const Int8Common* common);

    static Execution* createUnit(const Tensor* input, const Tensor* output, const MNN::Op* op, Backend* bn,
                                 const Int8Common* common, const float* bias, size_t biasSize);
};
} // namespace MNN

#endif /* ConvolutionIntFactory_hpp */
