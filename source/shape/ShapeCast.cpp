//
//  ShapeCast.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {

class CastSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto output = outputs[0];
        auto input  = inputs[0];
        TensorUtils::copyShape(input, output);

        const auto opParam = op->main_as_CastParam();
        outputs[0]->setType(opParam->dstT());

        return true;
    }
};

REGISTER_SHAPE(CastSizeComputer, OpType_Cast);

} // namespace MNN
