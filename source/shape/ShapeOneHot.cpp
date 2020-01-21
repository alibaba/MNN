//
//  ShapeOneHot.cpp
//  MNN
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#include "core/Macro.h"
#include "core/SizeComputer.hpp"

namespace MNN {
class ShapeOneHot : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(4 == inputs.size());
        auto indices     = inputs[0];
        auto depthTensor = inputs[1];

        const int depth = depthTensor->host<int>()[0];
        if (depth < 0) {
            return false;
        }
        const int indicesDimension = indices->dimensions();
        const int outputDimension  = indicesDimension + 1;

        auto param = op->main_as_OneHotParam();
        MNN_CHECK(param->dType() == DataType_DT_FLOAT, "TODO, support other data type!");
        int axis = param->axis();
        if (axis == -1) {
            axis = outputDimension + axis;
        }
        auto output                 = outputs[0];
        output->buffer().dimensions = outputDimension;
        output->buffer().type = inputs[2]->buffer().type;
        for (int i = 0; i < outputDimension; ++i) {
            if (i < axis) {
                output->setLength(i, indices->length(i));
            } else if (i == axis) {
                output->setLength(i, depth);
            } else {
                output->setLength(i, indices->length(i - 1));
            }
        }
        TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        return true;
    }
};

REGISTER_SHAPE_INPUTS(ShapeOneHot, OpType_OneHot, (std::vector<int>{1}));
} // namespace MNN
