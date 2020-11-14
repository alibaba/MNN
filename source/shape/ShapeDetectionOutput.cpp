//
//  ShapeDetectionOutput.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
namespace MNN {

// Size Computer
class DetectionOutputComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(3 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());

        // set dims
        auto &output    = outputs[0]->buffer();
        auto maxNumber = op->main_as_DetectionOutput()->keepTopK();

        output.dim[0].extent = 1;
        output.dim[1].extent = 1;
        output.dim[2].extent = maxNumber;
        output.dim[3].extent = 6; // maximum width
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        output.type = halide_type_of<float>();

        return true;
    }
};

REGISTER_SHAPE(DetectionOutputComputer, OpType_DetectionOutput);
} // namespace MNN
