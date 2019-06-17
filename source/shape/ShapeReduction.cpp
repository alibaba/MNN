//
//  ShapeReduction.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {
class ReductionComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto output                                       = outputs[0];
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        auto reduce                                       = op->main_as_ReductionParam();
        output->setType(reduce->dType());
        if (nullptr == reduce->dim() && inputs.size() == 1) {
            output->buffer().dimensions = 0;
            return true;
        }
        std::set<int> reduceDimSet;
        if (nullptr != reduce->dim()) {
            for (int i = 0; i < reduce->dim()->size(); ++i) {
                reduceDimSet.insert(reduce->dim()->data()[i]);
            }
        } else {
            auto input1 = inputs[1];
            auto size   = input1->elementSize();
            auto dims   = input1->host<int32_t>();
            for (int i = 0; i < size; ++i) {
                reduceDimSet.insert(dims[i]);
            }
        }

        auto input                = inputs[0];
        const int inputDimensions = input->dimensions();
        if (reduceDimSet.find(-1) != reduceDimSet.end()) {
            // dim set have -1 which mean applying reduction on last dimension
            reduceDimSet.erase(-1);
            reduceDimSet.insert(inputDimensions - 1);
        }

        std::vector<int> newDims;
        for (int i = 0; i < inputDimensions; ++i) {
            if (reduceDimSet.find(i) == reduceDimSet.end()) {
                newDims.push_back(input->length(i));
            } else if (reduce->keepDims()) {
                newDims.push_back(1);
            }
        }
        output->buffer().dimensions = (int)newDims.size();
        for (int i = 0; i < newDims.size(); ++i) {
            output->buffer().dim[i].extent = newDims[i];
            output->buffer().dim[i].flags  = 0;
        }

        return true;
    }
};

REGISTER_SHAPE(ReductionComputer, OpType_Reduction);
} // namespace MNN
