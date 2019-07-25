//
//  ShapeRNNSequenceGRU.cpp
//  MNN
//
//  Created by MNN on 2019/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {

class RNNSequenceGRUComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 <= outputs.size());

        auto input  = inputs[0];
        auto output = outputs[0];
        MNN_ASSERT(3 == input->dimensions());

        const auto rnnParam     = op->main_as_RNNParam();
        const int numUnits      = rnnParam->numUnits();
        bool keepAllOuptuts     = rnnParam->keepAllOutputs();
        bool isBidirectionalRNN = rnnParam->isBidirectionalRNN();
        MNN_ASSERT(2 == rnnParam->fwGateWeight()->dims()->size());
        MNN_ASSERT(2 * numUnits == rnnParam->fwGateWeight()->dims()->data()[1]);
        MNN_ASSERT((input->length(2) + numUnits) == rnnParam->fwGateWeight()->dims()->data()[0]);
        if (keepAllOuptuts) {
            TensorUtils::copyShape(input, output);
            output->setLength(2, rnnParam->numUnits());
            output->buffer().type = input->buffer().type;

            if (isBidirectionalRNN) {
                MNN_ASSERT(2 == outputs.size());
                auto outputBW = outputs[1];
                TensorUtils::copyShape(input, outputBW);
                outputBW->setLength(2, rnnParam->numUnits());
                outputBW->buffer().type = input->buffer().type;
            }
        } else {
            auto& inputBuffer          = input->buffer();
            auto& outputBuffer         = output->buffer();
            outputBuffer.dimensions    = 2;
            outputBuffer.dim[0].extent = inputBuffer.dim[0].extent;
            outputBuffer.dim[1].extent = rnnParam->numUnits();
            outputBuffer.type          = inputBuffer.type;

            if (isBidirectionalRNN) {
                MNN_ASSERT(2 == outputs.size());
                auto outputBW                = outputs[1];
                auto& outputBWBuffer         = outputBW->buffer();
                outputBWBuffer.dimensions    = 2;
                outputBWBuffer.dim[0].extent = inputBuffer.dim[0].extent;
                outputBWBuffer.dim[1].extent = rnnParam->numUnits();
                outputBWBuffer.type          = inputBuffer.type;
            }
        }

        return true;
    }
};

REGISTER_SHAPE(RNNSequenceGRUComputer, OpType_RNNSequenceGRU);
} // namespace MNN
