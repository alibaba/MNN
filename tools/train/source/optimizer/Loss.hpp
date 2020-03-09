//
//  Loss.hpp
//  MNN
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Loss_hpp
#define Loss_hpp

#include <MNN/expr/ExprCreator.hpp>

namespace MNN {
namespace Train {

MNN_PUBLIC Express::VARP _CrossEntropy(Express::VARP predicts, Express::VARP oneHotTargets);

MNN_PUBLIC Express::VARP _KLDivergence(Express::VARP predicts, Express::VARP oneHotTargets);

MNN_PUBLIC Express::VARP _MSE(Express::VARP predicts, Express::VARP oneHotTargets);

MNN_PUBLIC Express::VARP _MAE(Express::VARP predicts, Express::VARP oneHotTargets);

MNN_PUBLIC Express::VARP _Hinge(Express::VARP predicts, Express::VARP oneHotTargets);

MNN_PUBLIC Express::VARP _DistillLoss(Express::VARP studentLogits, Express::VARP teacherLogits, Express::VARP oneHotTargets,
                                                                const float temperature, const float alpha);

} // namespace Train
} // namespace MNN

#endif // Loss_hpp
