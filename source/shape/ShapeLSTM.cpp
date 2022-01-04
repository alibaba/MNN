//
//  ShapeLSTM.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// Size Computer
class LSTMComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        if (1 == outputs.size()) {
            // For compability for old version model
            MNN_ASSERT(1 == outputs.size());

            // copy dims
            auto &input  = inputs[0]->buffer();
            auto &output = outputs[0]->buffer();
            memcpy(output.dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);

            auto LSTM            = op->main_as_LSTM();
            output.dimensions = 4;
            output.dim[3].extent = LSTM->outputCount();
            output.dim[2].extent = 1;
            output.type = halide_type_of<float>();
            TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            return true;
        }
        // Onnx's LSTM
        MNN_ASSERT(inputs.size() >= 4);
        MNN_ASSERT(outputs.size() == 3);
        auto X = inputs[0];
        auto seqLength = X->length(0);
        auto batchSize = X->length(1);
        auto hiddenSize = op->main_as_LSTM()->outputCount();

        auto Y = outputs[0];
        auto ht = outputs[1];
        auto ct = outputs[2];
        Y->buffer().dimensions = 4;
        ht->buffer().dimensions = 3;
        ct->buffer().dimensions = 3;
        Y->setLength(0, seqLength);
        
        int direction = inputs[1]->length(0);
        MNN_ASSERT(1 == direction || 2 == direction);
        Y->setLength(1, direction);
        Y->setLength(2, batchSize);
        Y->setLength(3, hiddenSize);
        
        ht->setLength(0, direction);
        ht->setLength(1, batchSize);
        ht->setLength(2, hiddenSize);

        ct->setLength(0, direction);
        ct->setLength(1, batchSize);
        ct->setLength(2, hiddenSize);

        TensorUtils::getDescribe(Y)->dimensionFormat = TensorUtils::getDescribe(X)->dimensionFormat;
        TensorUtils::getDescribe(ht)->dimensionFormat = TensorUtils::getDescribe(X)->dimensionFormat;
        TensorUtils::getDescribe(ct)->dimensionFormat = TensorUtils::getDescribe(X)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(LSTMComputer, OpType_LSTM);

// LSTMCellBlock Size Computer
class LSTMBlockCellComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(inputs.size() == 8);
        MNN_ASSERT(outputs.size() == 7);
        for (int i = 0; i < outputs.size(); i++) {
            TensorUtils::copyShape(inputs[1], outputs[i]);
        }
        return true;
    }
};

REGISTER_SHAPE(LSTMBlockCellComputer, OpType_LSTMBlockCell);

// Size Computer
class RNNComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(inputs.size() >= 4 && outputs.size() == 2);
        
        auto X = inputs[0];
        auto seqLength = X->length(0), batchSize = X->length(1);
        auto hiddenSize = op->main_as_LSTM()->outputCount();

        auto Y = outputs[0], ht = outputs[1];
        Y->buffer().dimensions = 4;
        ht->buffer().dimensions = 3;
        Y->setLength(0, seqLength);
        
        int direction = inputs[1]->length(0);
        MNN_ASSERT(1 == direction || 2 == direction);
        Y->setLength(1, direction);
        Y->setLength(2, batchSize);
        Y->setLength(3, hiddenSize);
        
        ht->setLength(0, direction);
        ht->setLength(1, batchSize);
        ht->setLength(2, hiddenSize);

        TensorUtils::getDescribe(Y)->dimensionFormat = TensorUtils::getDescribe(X)->dimensionFormat;
        TensorUtils::getDescribe(ht)->dimensionFormat = TensorUtils::getDescribe(X)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(RNNComputer, OpType_RNN);

} // namespace MNN
