//
//  ShapeSplitGelu.cpp
//  MNN
//
//  Created by MNN on 2023/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class SplitGeLUSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input0 = inputs[0], output0 = outputs[0];
        MNN_ASSERT(inputs.size() == 1);
        MNN_ASSERT(input0->buffer().dimensions == 3);
	    output0->buffer().dimensions = 3;
        output0->buffer().dim[0].extent = input0->buffer().dim[0].extent;
        output0->buffer().dim[1].extent = input0->buffer().dim[1].extent;
        output0->buffer().dim[2].extent = input0->buffer().dim[2].extent / 2;
        //MNN_PRINT("SplitGelu shape:%d %d, %d %d %d %d %d\n", input0->buffer().dimensions, output0->buffer().dimensions, input0->buffer().dim[0].extent, input0->buffer().dim[1].extent, input0->buffer().dim[2].extent, input0->buffer().dim[3].extent, input0->buffer().dim[4].extent);
        output0->buffer().type = input0->buffer().type;
        TensorUtils::getDescribe(output0)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;

        return true;
    }
};

class SeqLen2SpatialSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 4);
        auto input3 = inputs[0];
        auto output0 = outputs[0];
	    output0->buffer().dimensions = input3->buffer().dimensions;
        for(int i = 0; i < input3->buffer().dimensions; i++) {
            output0->buffer().dim[i].extent = input3->buffer().dim[i].extent;
        }
        output0->buffer().type = input3->buffer().type;
        TensorUtils::getDescribe(output0)->dimensionFormat = TensorUtils::getDescribe(input3)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(SplitGeLUSizeComputer, OpType_SplitGeLU);
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(SeqLen2SpatialSizeComputer, OpType_SeqLen2Spatial);

} // namespace MNN
#endif
