//
//  ShapeRank.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {

class RankComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // output is Scalar
        outputs[0]->buffer().dimensions = 0;
        outputs[0]->setType(MNN::DataType_DT_INT32);
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE(RankComputer, OpType_Rank);
} // namespace MNN
