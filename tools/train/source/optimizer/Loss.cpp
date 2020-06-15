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

Express::VARP _DistillLoss(Express::VARP studentLogits, Express::VARP teacherLogits, Express::VARP oneHotTargets, const float temperature, const float alpha) {
    auto info = teacherLogits->getInfo();
    if (info->order == NC4HW4) {
        teacherLogits = _Convert(teacherLogits, NCHW);
        studentLogits = _Convert(studentLogits, NCHW);
    }
    MNN_ASSERT(studentLogits->getInfo()->dim.size() == 2);
    MNN_ASSERT(studentLogits->getInfo()->dim == teacherLogits->getInfo()->dim);
    MNN_ASSERT(studentLogits->getInfo()->dim == oneHotTargets->getInfo()->dim);
    MNN_ASSERT(alpha >= 0 && alpha <= 1);
    auto softTargets = _Softmax(teacherLogits * _Scalar(1 / temperature));
    auto studentPredict = _Softmax(studentLogits * _Scalar(1 / temperature));
    auto loss1 = _Scalar(temperature * temperature) * _KLDivergence(studentPredict, softTargets);
    auto loss2 = _CrossEntropy(_Softmax(studentLogits), oneHotTargets);
    auto loss = _Scalar(alpha) * loss1 + _Scalar(1 - alpha) * loss2;
    return loss;
}

} // namespace Train
} // namespace MNN
