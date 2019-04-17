//
//  ShapeProposal.cpp
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
class ProposalComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(3 == inputs.size());
        MNN_ASSERT(1 <= outputs.size() && outputs.size() <= 2);

        auto proposal        = op->main_as_Proposal();
        auto &output         = outputs[0]->buffer();
        output.dim[3].extent = 1;
        output.dim[2].extent = 1;
        output.dim[1].extent = 5;
        output.dim[0].extent = proposal->afterNmsTopN() * inputs[0]->buffer().dim[0].extent;

        if (outputs.size() > 1) {
            auto &roi         = outputs[1]->buffer();
            roi.dim[3].extent = 1;
            roi.dim[2].extent = 1;
            roi.dim[1].extent = 1;
            roi.dim[0].extent = proposal->afterNmsTopN() * inputs[0]->buffer().dim[0].extent;
        }

        return true;
    }
};

REGISTER_SHAPE(ProposalComputer, OpType_Proposal);
} // namespace MNN
