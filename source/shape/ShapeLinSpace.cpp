//
//  ShapeLinSpace.cpp
//  MNN
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class LinSpaceSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 3);
        MNN_ASSERT(outputs.size() == 1);
        auto& ib1 = inputs[0]->buffer();
        auto& ib2 = inputs[1]->buffer();
        auto& ib3 = inputs[2]->buffer();
        auto& ob = outputs[0]->buffer();
        MNN_ASSERT(ib1.dimensions == 0);
        MNN_ASSERT(ib2.dimensions == 0);
        MNN_ASSERT(ib3.dimensions == 0);

        MNN_ASSERT(inputs[0]->getType() == halide_type_of<float>());
        MNN_ASSERT(inputs[1]->getType() == halide_type_of<float>());
        MNN_ASSERT(inputs[2]->getType() == halide_type_of<int32_t>());

        int num = inputs[2]->host<int32_t>()[0];
        MNN_ASSERT(num > 0);
        
        ob.dimensions = 1;
        ob.dim[0].extent = num;
        outputs[0]->setType(DataType_DT_FLOAT);
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS(LinSpaceSizeComputer, OpType_LinSpace, {2});
} // namespace MNN
