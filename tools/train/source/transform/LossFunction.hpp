//
//  LossFunction.hpp
//  MNN
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LossFunction_hpp
#define LossFunction_hpp

#include <map>
#include <memory>
#include "OpConverter.hpp"
#include "Tensor.hpp"
class MNN_PUBLIC LossFunction {
public:
    static MNN::OpT* addSubEclLoss(MNN::NetT* net, const MNN::OpT* compare,
                                   std::map<int, std::shared_ptr<MNN::Tensor>>& tensors);
    static MNN::OpT* addProbLoss(MNN::NetT* net, const MNN::OpT* compare,
                                 std::map<int, std::shared_ptr<MNN::Tensor>>& tensors);
};

#endif /* LossFunction_hpp */
