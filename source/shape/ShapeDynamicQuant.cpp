//
//  ShapeDet.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class DynamicQuantComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(outputs.size() == 3);
        if (inputs.size() != 1) {
            MNN_ERROR("DynamicQuant only accept 1 input\n");
            return false;
        }
        auto input = inputs[0];
        auto output = outputs[0];
        int dimSize = input->dimensions();
        output->buffer().dimensions = dimSize;
        for (int i = 0; i < dimSize; ++i) {
            output->buffer().dim[i].extent = input->buffer().dim[i].extent;
        }
        auto scale = outputs[1];
        auto zeroPoint = outputs[2];
        scale->buffer().dimensions = 1;
        zeroPoint->buffer().dimensions = 1;
        scale->buffer().dim[0].extent = 1;
        zeroPoint->buffer().dim[0].extent = 1;
        
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        output->buffer().type = halide_type_of<int8_t>();
        
        TensorUtils::getDescribe(scale)->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        scale->buffer().type = halide_type_of<float>();
        
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        zeroPoint->buffer().type = halide_type_of<float>();

        return true;
    }
};

REGISTER_SHAPE(DynamicQuantComputer, OpType_DynamicQuant);

} // namespace MNN
