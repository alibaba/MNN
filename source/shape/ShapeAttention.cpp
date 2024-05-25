//
//  ShapeAttention.cpp
//  MNN
//
//  Created by MNN on 2023/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"


namespace MNN {
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

class FmhaV2SizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input0 = inputs[0], output0 = outputs[0];
        MNN_ASSERT(inputs.size() == 1);
        MNN_ASSERT(input0->buffer().dimensions == 3);

        output0->buffer().dim[0].extent = input0->buffer().dim[0].extent;
        output0->buffer().dim[1].extent = input0->buffer().dim[1].extent;
        output0->buffer().dim[2].extent = input0->buffer().dim[2].extent/3;
        output0->buffer().dimensions = 3;
        //MNN_PRINT("fmhaV2 shape:%d %d, %d %d %d %d %d\n", input0->buffer().dimensions, output0->buffer().dimensions, input0->buffer().dim[0].extent, input0->buffer().dim[1].extent, input0->buffer().dim[2].extent, input0->buffer().dim[3].extent, input0->buffer().dim[4].extent);
        //MNN_ASSERT(input0->buffer().dim[3].extent == 3);
        output0->buffer().type = input0->buffer().type;
        TensorUtils::getDescribe(output0)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;
        //printf("fmhaV2 shape:%d %d, %d %d %d\n", input0->buffer().dimensions, output0->buffer().dimensions, input0->buffer().dim[0].extent, input0->buffer().dim[1].extent, input0->buffer().dim[2].extent);
        return true;
    }
};

class FmhcaSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(outputs.size() == 1);
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        auto output0 = outputs[0];
        MNN_ASSERT(input0->buffer().dimensions == 3);
        MNN_ASSERT(input1->buffer().dimensions == 3);

        output0->buffer().dim[0].extent = input0->buffer().dim[0].extent;
        output0->buffer().dim[1].extent = input0->buffer().dim[1].extent;
        output0->buffer().dim[2].extent = input0->buffer().dim[2].extent;
        output0->buffer().dimensions = 3;
        //MNN_ASSERT(input1->buffer().dim[0].extent == input0->buffer().dim[0].extent);
        //MNN_ASSERT(input1->buffer().dim[2].extent == input0->buffer().dim[2].extent);
        //MNN_ASSERT(input1->buffer().dim[4].extent == input0->buffer().dim[3].extent);
        output0->buffer().type = input0->buffer().type;
        TensorUtils::getDescribe(output0)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;
        //printf("fmhca shape:%d %d %d, %d %d %d\n", input0->buffer().dimensions, input1->buffer().dimensions, output0->buffer().dimensions, input0->buffer().dim[0].extent, input0->buffer().dim[1].extent, input0->buffer().dim[2].extent);
        return true;
    }
};

class AttentionSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input = inputs[0], output = outputs[0];
        MNN_ASSERT(input->buffer().dimensions == 4);
        output->buffer().dim[0].extent = input->buffer().dim[0].extent;
        output->buffer().dim[1].extent = input->buffer().dim[1].extent;
        output->buffer().dim[2].extent = input->buffer().dim[2].extent * input->buffer().dim[3].extent;
        output->buffer().dimensions = 3;
        output->buffer().type = input->buffer().type;
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        return true;
    }
};


REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(FmhaV2SizeComputer, OpType_FmhaV2);
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(FmhcaSizeComputer, OpType_Fmhca);
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(AttentionSizeComputer, OpType_Attention);
#endif

} // namespace MNN

