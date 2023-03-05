//
//  ShapeSliceTf.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

class SliceTfComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 3);
        MNN_ASSERT(outputs.size() == 1);

        auto input = inputs[0];
        // these two inputs should be const
        auto begin_tensor = inputs[1];
        auto size_tensor  = inputs[2];

        MNN_ASSERT(begin_tensor->buffer().dimensions == 1);
        MNN_ASSERT(size_tensor->buffer().dimensions == 1);
        MNN_ASSERT(input->buffer().dimensions >= 1);
        MNN_ASSERT(input->buffer().dimensions == begin_tensor->buffer().dim[0].extent);
        MNN_ASSERT(input->buffer().dimensions == size_tensor->buffer().dim[0].extent);

        auto output                 = outputs[0];
        output->buffer().dimensions = input->buffer().dimensions;
        output->buffer().type       = input->buffer().type;
        int dim                     = 0;
        auto sizePtr = size_tensor->host<int32_t>();
        for (int i = 0; i < input->buffer().dimensions; i++) {
            dim = sizePtr[i];
            if (dim == -1 ) {
                auto begin = begin_tensor->host<int32_t>()[i];
                if (begin < 0) {
                    begin += input->length(i);
                }
                dim = input->buffer().dim[i].extent - begin;
            }
            MNN_ASSERT(dim <= input->length(i));
            output->buffer().dim[i].extent = dim;
        }
        for (int i=0; i<outputs.size(); ++i) {
            TensorUtils::getDescribe(outputs[i])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(SliceTfComputer, OpType_SliceTf, (std::vector<int>{1, 2}));
} // namespace MNN
