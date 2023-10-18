//
//  ShapeReduction.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
static int _getRealAxis(int axis, int n) {
    if (axis < 0) {
        return axis + n;
    }
    return axis;
}
class ReductionComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto output                                       = outputs[0];
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        auto reduce                                       = op->main_as_ReductionParam();
        output->buffer().type = inputs[0]->buffer().type;
        if (nullptr == reduce->dim() && inputs.size() == 1) {
            if (reduce->keepDims()) {
                output->buffer().dimensions = inputs[0]->dimensions();
                for (int i = 0; i < inputs[0]->dimensions(); i++) {
                    output->setLength(i, 1);
                }
            } else {
                output->buffer().dimensions = 0;
            }
            return true;
        }
        uint8_t reduceMask[MNN_MAX_TENSOR_DIM];
        ::memset(reduceMask, 0, sizeof(uint8_t) * MNN_MAX_TENSOR_DIM);
        if (nullptr != reduce->dim()) {
            for (int i = 0; i < reduce->dim()->size(); ++i) {
                reduceMask[_getRealAxis(reduce->dim()->data()[i], inputs[0]->dimensions())] = 1;
            }
        } else {
            auto input1 = inputs[1];
            auto size   = input1->elementSize();
            auto dims   = input1->host<int32_t>();
            for (int i = 0; i < size; ++i) {
                reduceMask[_getRealAxis(dims[i], inputs[0]->dimensions())] = 1;
            }
        }

        auto input                = inputs[0];
        const int inputDimensions = input->dimensions();

        int offset = 0;
        for (int i = 0; i < inputDimensions; ++i) {
            if (1 == reduceMask[i]) {
                if (reduce->keepDims()) {
                    output->buffer().dim[offset].extent = 1;
                    offset++;
                }
                continue;
            }
            output->buffer().dim[offset].extent = input->length(i);
            offset++;
        }
        output->buffer().dimensions = offset;
        return true;
    }
};

REGISTER_SHAPE_INPUTS(ReductionComputer, OpType_Reduction, {1});
} // namespace MNN
