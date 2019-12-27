//
//  Transformer.hpp
//  MNN
//
//  Created by MNN on 2019/12/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Transformer_hpp
#define Transformer_hpp
#include <MNN/expr/Optimizer.hpp>

namespace MNN {
namespace Train {
class MNN_PUBLIC Transformer {
public:
    struct TrainConfig {
        std::vector<std::string> variableLimits;
    };

    static std::shared_ptr<Express::Optimizer> turnModelToTrainable(TrainConfig config);
    static std::shared_ptr<Express::Optimizer> turnModelToInfer();
};
} // namespace Train
} // namespace MNN
#endif
