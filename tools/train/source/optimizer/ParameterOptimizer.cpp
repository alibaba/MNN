//
//  ParameterOptimizer.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ParameterOptimizer.hpp"

namespace MNN {
namespace Train {

bool ParameterOptimizer::step(Express::VARP loss) {
    auto res = this->onGetNextParameter(loss);
    for (auto iter : res) {
        iter.second.fix(Express::VARP::CONST);
    }
    for (auto iter : res) {
        Express::Variable::replace(iter.first, iter.second);
    }
    return !res.empty();
}

} // namespace Train
} // namespace MNN
