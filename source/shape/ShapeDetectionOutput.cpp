//
//  ShapeDetectionOutput.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
namespace MNN {

// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Size Computer
// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class DetectionOutputComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(3 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());

        // set dims
        auto &priorbox  = inputs[2]->buffer();
        auto &output    = outputs[0]->buffer();
        auto priorCount = priorbox.dim[2].extent / 4;

        output.dim[0].extent = 1;
        output.dim[1].extent = 1;
        output.dim[2].extent = priorCount;
        output.dim[3].extent = 6; // maximum width

        return true;
    }
};

REGISTER_SHAPE(DetectionOutputComputer, OpType_DetectionOutput);
} // namespace MNN
