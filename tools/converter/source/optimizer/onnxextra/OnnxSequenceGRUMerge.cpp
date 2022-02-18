//
//  OnnxSequenceGRUMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2021/03/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

VARP _SliceConst(VARP x, const std::vector<int>& starts, const std::vector<int>& sizes) {
    auto startVAR = _Const((const void*)starts.data(), {static_cast<int>(starts.size())}, NHWC, halide_type_of<int>());
    auto sizeVAR = _Const((const void*)sizes.data(), {static_cast<int>(sizes.size())}, NHWC, halide_type_of<int>());
    return _Slice(x, startVAR, sizeVAR);
}

class OnnxSequenceGRUTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        if (inputs.size() < 4 || inputs[3].get() == nullptr) { // X W R B
            MNN_ERROR("Don't support optional 4th input (B)\n");
            return nullptr;
        }
        auto rnnGRUParam = new MNN::RNNParamT;
        std::unique_ptr<OpT> gru(new OpT);
        gru->name       = expr->name();
        gru->type       = OpType_RNNSequenceGRU;
        gru->main.type  = OpParameter_RNNParam;
        gru->main.value = rnnGRUParam;

        auto extra = expr->get()->main_as_Extra();
        auto attr  = extra->attr();
        if (nullptr != attr) {
            for (int i = 0; i < attr->size(); ++i) {
                auto attUnit = attr->GetAs<Attribute>(i);
                if (attUnit->key()->str() == "hidden_size") {
                    rnnGRUParam->numUnits = static_cast<int32_t>(attUnit->i());
                } else if(attUnit->key()->str() == "direction") {

                    rnnGRUParam->isBidirectionalRNN = attUnit->s()->str() == "bidirectional";
                } else if (attUnit->key()->str() == "linear_before_reset") {
                    rnnGRUParam->linearBeforeReset = static_cast<bool>(attUnit->i());
                }

            }
        }

        rnnGRUParam->keepAllOutputs = true;
        // In onnx, the final hidden output(Y_h) and hidden in different sequencial(Y) could be outputed both,
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#outputs-0---2
        // todo: detect the next op in DFG, if Y is never used, assign mKeepAllOutputs as false.


        auto W_rzh = inputs[1];
        auto R_rzh = inputs[2];
        auto B_2rzh = inputs[3];
        auto W_info = W_rzh->getInfo();
        auto R_info = R_rzh->getInfo();
        auto B_info = B_2rzh->getInfo();
        auto hiddenSize = rnnGRUParam->numUnits;
        auto inputSize = W_info->dim[2];
        LOG(INFO) << " OnnxSequenceGRUMerge:  W shape:{"
                  << W_info->dim[0] << ", " << W_info->dim[1] << ", " << W_info->dim[2] << "}; R shape: {"
                  << R_info->dim[0] << ", " << R_info->dim[1] << ", " << R_info->dim[2]
                  << "}, inputs num:" << inputs.size() <<  ", outputs num:" << expr->outputSize();

        if (nullptr == B_info || nullptr == B_2rzh->readMap<float>()) {
            MNN_ERROR("Can't solve GRU because bias is not const\n");
            return nullptr;
        }

        MNN_ASSERT(3 * hiddenSize == W_info->dim[1]);
        MNN_ASSERT(3 * hiddenSize == R_info->dim[1]);
        MNN_ASSERT(hiddenSize == R_info->dim[2]);
        MNN_ASSERT(rnnGRUParam->isBidirectionalRNN + 1 == W_info->dim[0]);

        const int forwardParamNumber = 5;
        std::vector<VARP> gruInput(1 + forwardParamNumber * (rnnGRUParam->isBidirectionalRNN + 1));
        gruInput[0] = inputs[0];

        auto W_R = _Concat({W_rzh, R_rzh}, 2);

        // forward gru
        auto forward_W_R = _Squeeze(_SliceConst(W_R, {0, 0, 0}, {1, 3 * hiddenSize, inputSize + hiddenSize}), {0});
        forward_W_R = _Transpose(forward_W_R, {1, 0});
        gruInput[1] = _SliceConst(forward_W_R, {0, 0}, {inputSize + hiddenSize , 2 * hiddenSize}); // gateWeight
        gruInput[3] = _SliceConst(forward_W_R, {0, 2 * hiddenSize}, {inputSize + hiddenSize, hiddenSize}); // candidateWeight

        auto forward_B = _SliceConst(B_2rzh, {0, 0}, {1, 6 * hiddenSize});
        gruInput[2] = _SliceConst(forward_B, {0, 0}, {1, 2 * hiddenSize}); // gateBias
        gruInput[4] = _SliceConst(forward_B, {0, 2 * hiddenSize}, {1, hiddenSize});// candidateBias
        gruInput[5] = _SliceConst(forward_B, {0, 3 * hiddenSize}, {1, 3 * hiddenSize});// recurrentBias


        // backward gru
        if(rnnGRUParam->isBidirectionalRNN) {
            auto backward_W_R = _Squeeze(_SliceConst(W_R, {1, 0, 0}, {1, 3 * hiddenSize, inputSize + hiddenSize}), {0});
            backward_W_R = _Transpose(backward_W_R, {1, 0});
            gruInput[6] = _SliceConst(backward_W_R, {0, 0}, {inputSize + hiddenSize , 2 * hiddenSize}); // backward gateWeight
            gruInput[8] = _SliceConst(backward_W_R, {0, 2 * hiddenSize}, {inputSize + hiddenSize, hiddenSize}); //backward candidateWeight
            auto backward_B = _SliceConst(B_2rzh, {1, 0}, {1, 6 * hiddenSize});
            gruInput[7] = _SliceConst(backward_B, {0, 0}, {1, 2 * hiddenSize}); // backward gateBias
            gruInput[9] = _SliceConst(backward_B, {0, 2 * hiddenSize}, {1, hiddenSize});// backward candidateBias
            gruInput[10] = _SliceConst(backward_B, {0, 3 * hiddenSize}, {1, 3 * hiddenSize});// backward recurrentBias
        }

        // auto sequence_lens = inputs[4]; sequence_lens is ommitted at onnxConverter.cpp
        if (inputs.size() > 4 && inputs[4].get() != nullptr) {
            MNN_ERROR("Don't support sequence_lens input, all batch have seq_length\n");
            return nullptr;
        }
        if (inputs.size() > 5) { // initial_h exist, shape is [num_directions, batch_size, hidden_size]
            gruInput.push_back(inputs[5]);
        }

        auto gruExpr = Expr::create(gru.get(), gruInput, expr->outputSize());
        gruExpr->setName(expr->name());
        for (int i = 0; i < expr->outputSize(); ++i) {
            Variable::create(gruExpr, i)->setName(expr->outputName(i));
        }
        return gruExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("GRU", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSequenceGRUTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
