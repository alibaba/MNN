//
//  ShapeUnpack.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

class UnpackComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        auto unpack    = op->main_as_Axis();
        const int axis = unpack->axis();

        auto &input = inputs[0]->buffer();

        const int inputDimensions = input.dimensions;
        MNN_ASSERT(1 <= inputDimensions);

        std::vector<int> outDims;
        for (int i = 0; i < inputDimensions; i++) {
            if (axis == i) {
                continue;
            }
            outDims.push_back(input.dim[i].extent);
        }
        const int outputDimensions = inputDimensions - 1;
        MNN_ASSERT(outDims.size() == outputDimensions);
        for (int i = 0; i < outputs.size(); i++) {
            auto &output      = outputs[i]->buffer();
            output.dimensions = outputDimensions;
            output.type       = input.type;
            for (int j = 0; j < outputDimensions; j++) {
                output.dim[j].extent = outDims[j];
            }
        }

        return true;
    }
};

REGISTER_SHAPE(UnpackComputer, OpType_Unpack);
} // namespace MNN
