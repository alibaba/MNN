//
//  ShapeReduction.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {
class ReductionComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto output = outputs[0];
        auto reduce = op->main_as_ReductionParam();
        output->setType(reduce->dType());
        if (nullptr == reduce->dim()) {
            output->buffer().dimensions = 0;
            return true;
        }
        std::set<int> reduceDimSet;
        for (int i = 0; i < reduce->dim()->size(); ++i) {
            reduceDimSet.insert(reduce->dim()->data()[i]);
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
