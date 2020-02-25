//
//  Loss.cpp
//  MNN
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Loss.hpp"

using namespace MNN::Express;

namespace MNN {
namespace Train {

Express::VARP _CrossEntropy(Express::VARP predicts, Express::VARP oneHotTargets) {
    MNN_ASSERT(predicts->getInfo()->dim.size() == 2);
    MNN_ASSERT(predicts->getInfo()->dim == oneHotTargets->getInfo()->dim);
    auto loss = _Negative(_ReduceMean(_ReduceSum(_Log(predicts) * oneHotTargets, {1}), {}));
    return loss;
}

Express::VARP _KLDivergence(Express::VARP predicts, Express::VARP oneHotTargets) {
    MNN_ASSERT(predicts->getInfo()->dim.size() == 2);
    MNN_ASSERT(predicts->getInfo()->dim == oneHotTargets->getInfo()->dim);
    auto loss = _ReduceMean(_ReduceSum(_Multiply(predicts, _Log(predicts) - _Log(oneHotTargets)), {1}), {});
    return loss;
}

Express::VARP _MSE(Express::VARP predicts, Express::VARP oneHotTargets) {
    MNN_ASSERT(predicts->getInfo()->dim.size() == 2);
    MNN_ASSERT(predicts->getInfo()->dim == oneHotTargets->getInfo()->dim);
    auto loss = _ReduceMean(_ReduceSum(_Square(predicts - oneHotTargets), {1}), {});
    return loss;
}

Express::VARP _MAE(Express::VARP predicts, Express::VARP oneHotTargets) {
    MNN_ASSERT(predicts->getInfo()->dim.size() == 2);
    MNN_ASSERT(predicts->getInfo()->dim == oneHotTargets->getInfo()->dim);
    auto loss = _ReduceMean(_ReduceSum(_Abs(predicts - oneHotTargets), {1}), {});
    return loss;
}

Express::VARP _Hinge(Express::VARP predicts, Express::VARP oneHotTargets) {
    MNN_ASSERT(predicts->getInfo()->dim.size() == 2);
    MNN_ASSERT(predicts->getInfo()->dim == oneHotTargets->getInfo()->dim);
    auto loss = _ReduceMean(_ReduceSum(_Maximum(_Const(0.), _Const(1.) - predicts * oneHotTargets), {1}), {});
    return loss;
}

} // namespace Train
} // namespace MNN
