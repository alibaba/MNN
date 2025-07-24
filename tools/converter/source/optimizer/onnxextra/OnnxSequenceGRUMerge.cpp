//
//  OnnxSequenceGRUMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2021/03/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "OnnxRNNHelper.hpp"
namespace MNN {
namespace Express {
static VARP _SliceConst(VARP x, const std::vector<int>& starts, const std::vector<int>& sizes) {
    auto startVAR = _Const((const void*)starts.data(), {static_cast<int>(starts.size())}, NHWC, halide_type_of<int>());
    auto sizeVAR = _Const((const void*)sizes.data(), {static_cast<int>(sizes.size())}, NHWC, halide_type_of<int>());
    return _Slice(x, startVAR, sizeVAR);
}

static VARP _computeRecMain(VARP Gate, VARP HI, VARP R_zrh, VARP BR, VARP BRH, int hiddenSize, int direction) {
    VARP Z;
    VARP R;
    VARP HR = _makeConvForRStep(HI, R_zrh, hiddenSize, direction, BR);
    auto splitsR = _Split(HR, {3}, 1);
    auto splits = _Split(Gate, {3}, 1);
    /**
     zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)

     rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

     ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0

     ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0

     Ht = (1 - zt) (.) ht + zt (.) Ht-1 This operator has optional inputs/outputs. See the doc for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
     */
    Z = _Sigmoid(splits[0] + splitsR[0]);
    R = _Sigmoid(splits[1] + splitsR[1]);
    VARP H;
    if (nullptr == BRH) {
        H = _Tanh(splits[2] + R * splitsR[2]);
    } else {
        // rnnGRUParam->linearBeforeReset
        H = _Tanh(splits[2] + R * splitsR[2] - R * BRH + BRH);
    }
    H = H - Z *(H-HI);
    return H;
}

class OnnxSequenceGRUTransform : public OnnxExtraManager::Transform {
public:
    static EXPRP _turnGRU2While(OpT* gru, EXPRP expr) {
        auto inputs = expr->inputs();
        auto W_zrh = inputs[1];
        auto R_zrh = inputs[2];
        VARP B_2rzh = nullptr;
        if (inputs.size() >= 4 && inputs[3].get() != nullptr) { // X W R B
            B_2rzh = inputs[3];
        }
        VARP O_InitOrigin = nullptr;
        if (inputs.size() >= 6) {
            O_InitOrigin = inputs[5];
        }
        bool singleSeq = false;
        if (inputs[0]->getInfo() != nullptr && inputs[0]->getInfo()->dim[0] == 1) {
            singleSeq = true;
            MNN_PRINT("Single SeqLength, don't use while but unrool it\n");
        }
        VARP BW = nullptr;
        VARP BR = nullptr;
        VARP BRH = nullptr;
        auto rnnGRUParam = gru->main.AsRNNParam();
        int directionNumer = rnnGRUParam->isBidirectionalRNN ? 2 : 1;
        auto W_info = W_zrh->getInfo();
        auto R_info = R_zrh->getInfo();
        auto hiddenSize = rnnGRUParam->numUnits;
        auto inputSize = W_info->dim[2];
        if (nullptr != B_2rzh) {
            auto BSplit = _Split(B_2rzh, {2}, 1);
            BW = BSplit[0];
            BR = BSplit[1];
            if (!rnnGRUParam->linearBeforeReset) {
                BRH = _Split(BR, {3}, 1)[2];
                BRH = _Reshape(BRH, {1, hiddenSize, 1, 1}, NCHW);
                BRH.fix(VARP::CONSTANT);
            }
        }
        std::vector<VARP> O_InitGroup;
        if (nullptr == O_InitOrigin) {
            auto zeroInit = _Const(0.0f, std::vector<int>{1, hiddenSize, 1, 1}, NCHW);
            for (int i=0; i<directionNumer; ++i) {
                O_InitGroup.emplace_back(zeroInit);
            }
        } else {
            O_InitGroup = _Split(O_InitOrigin, {directionNumer}, 0);
            for (int i=0; i<directionNumer; ++i) {
                O_InitGroup[i] = _Reshape(O_InitGroup[i], {-1, hiddenSize, 1, 1}, NCHW);
            }
        }
        auto zero = _Unsqueeze(_Scalar<int32_t>(0), {0});
        auto one = _Unsqueeze(_Scalar<int32_t>(1), {0});
        auto negone = _Unsqueeze(_Scalar<int32_t>(-1), {0});
        // GRU Has three component: rzh
        auto componentVar = _Unsqueeze(_Scalar<int32_t>(3), {0});
        std::vector<VARP> Output;
        std::vector<VARP> OLast;
        auto inputShape = _Shape(inputs[0], true);
        auto seqLengthVar = _Slice(inputShape, _Unsqueeze(_Scalar<int32_t>(0), {0}), one);
        auto batchFullVar = _Slice(inputShape, _Unsqueeze(_Scalar<int32_t>(1), {0}), one);
        auto hiddenSizeVar = _Unsqueeze(_Scalar<int>(hiddenSize), {0});

        for (int i=0; i<directionNumer; ++i) {
            // FirstPart: Gate = MatMul(X, W, B) :  N * hiddenSize, seqLength * batchSize
            // Gate = Conv(Reshape(X, {seqLength * batch, inputSize, 1, 1}))
            // Gate: seqLength * batch, N * hiddenSize, 1, 1
            VARP FullGate = _makeConvForW(W_zrh, BW, inputs[0], inputSize, i);
            if (singleSeq) {
                auto H = _computeRecMain(FullGate, O_InitGroup[i], R_zrh, BR, BRH,hiddenSize, i);
                Output.emplace_back(_Unsqueeze(H, {0}));
                OLast.emplace_back(H);
                continue;
            }
            // Make SubGraph
            auto bodyGraphName = gru->name + "_main" + std::to_string(i);
            {
                auto inputShape = _Input({}, NCHW, halide_type_of<int>());
                inputShape->setName("inputshape");
                auto batchVar = _Slice(inputShape, _Unsqueeze(_Scalar<int32_t>(1), {0}), one);
                auto hiddenSizeVar = _Unsqueeze(_Scalar<int32_t>(hiddenSize), {0});
                
                auto step = _Input({}, NCHW, halide_type_of<int>());
                step->setName("i");
                VARP GateFull = _Input({-1, -1, 1, 1}, NC4HW4);
                GateFull->setName("Gate");
                auto size = _Concat({batchVar, hiddenSizeVar * componentVar, one, one}, 0);
                VARP start;
                if (0 == i) {
                    start = _Concat({batchVar * step, zero, zero, zero}, 0);
                } else {
                    auto seqLengthVar = _Slice(inputShape, _Unsqueeze(_Scalar<int32_t>(0), {0}), one);
                    start = _Concat({batchVar * (seqLengthVar - one - step), zero, zero, zero}, 0);
                }
                auto Gate = _Slice(GateFull, start, size);
                VARP Z;
                VARP R;
                VARP HI = _Input({-1, hiddenSize, 1, 1}, NCHW);
                HI->setName("H");
                auto H = _computeRecMain(Gate, HI, R_zrh, BR, BRH,hiddenSize, i);
                H->setName("O_next");
                auto cond = _Input({}, NCHW, halide_type_of<int>());
                cond->setName("cond");
                std::unique_ptr<OpT> copyOp(new OpT);
                copyOp->type = OpType_Identity;
                EXPRP copyExpr = Expr::create(copyOp.get(), {H}, 1);
                auto OCopy = Variable::create(copyExpr);
                OCopy->setName("O_next_copy");

                auto outputCond = _Scalar<float>(1.0f);
                outputCond->setName("output_cond");
                ExecutorScope::Current()->registerSubGraph(bodyGraphName, {outputCond, inputShape, GateFull, H, OCopy}, {step, cond, inputShape, GateFull, HI});
            }

            // Make Copy Op to fuse three varps
            std::unique_ptr<OpT> loopOp(new OpT);
            loopOp->type = OpType_While;
            loopOp->main.value = new WhileParamT;
            loopOp->main.type = OpParameter_WhileParam;
            auto whileP = loopOp->main.AsWhileParam();
            whileP->body_graph = bodyGraphName;
            auto cond = _Scalar<int>(1);
            auto whileInputs = std::vector<VARP>{seqLengthVar, cond, inputShape, FullGate, O_InitGroup[i]};
            auto whileExpr = Expr::create(loopOp.get(), whileInputs, 4);
            auto directionO = Variable::create(whileExpr, 3);
            if (1 == i) {
                directionO = _Reverse(directionO, _Scalar<int>(0));
            }
            Output.emplace_back(directionO);
            OLast.emplace_back(Variable::create(whileExpr, 2));
        }
        for (int i=0; i<directionNumer; ++i) {
            Output[i] = OnnxExtraManager::_ReshapeF(Output[i], _Concat({seqLengthVar, one, batchFullVar, hiddenSizeVar}, 0));
            OLast[i] = OnnxExtraManager::_ReshapeF(OLast[i], _Concat({one, batchFullVar, hiddenSizeVar}, 0));
        }
        std::unique_ptr<OpT> copyOp(new OpT);
        copyOp->type = OpType_Identity;
        EXPRP resultExpr;
        if (1 == directionNumer) {
            resultExpr = Expr::create(copyOp.get(), {Output[0], OLast[0]}, 2);
        } else {
            auto o0 = _Concat(Output, 1);
            auto o1 = _Concat(OLast, 0);
            resultExpr = Expr::create(copyOp.get(), {o0, o1}, 2);
        }
        resultExpr->setName(gru->name);
        return resultExpr;
    }
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
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


        auto W_zrh = inputs[1];
        auto R_zrh = inputs[2];
        VARP B_2rzh = nullptr;
        if (inputs.size() >= 4 && inputs[3].get() != nullptr) { // X W R B
            B_2rzh = inputs[3];
        }
        bool biasValid = B_2rzh == nullptr || B_2rzh->readMap<void>() != nullptr;
        auto config = Global<modelConfig>::Get();
        if (!config->useOriginRNNImpl) {
            if (W_zrh->readMap<void>() != nullptr && biasValid && R_zrh->readMap<void>() != nullptr) {
                MNN_PRINT("Try to use While to compute GRU for %s, if don't want it, add --useOriginRNNImpl \n", expr->name().c_str());
                return _turnGRU2While(gru.get(), expr);
            }
        }
        auto W_info = W_zrh->getInfo();
        auto R_info = R_zrh->getInfo();
        if (nullptr == W_info || nullptr == R_info) {
            MNN_ERROR("Don't GRU for not W / R's shape not valid\n");
            return nullptr;
        }
        auto hiddenSize = rnnGRUParam->numUnits;
        auto inputSize = W_info->dim[2];
        if (nullptr == B_2rzh) {
            int direction = rnnGRUParam->isBidirectionalRNN ? 2 : 1;
            B_2rzh = _Const(0.0f, {direction , 6 * hiddenSize}, NCHW);
        }

        if (nullptr != B_2rzh && nullptr == B_2rzh->readMap<float>()) {
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

        auto W_R = _Concat({W_zrh, R_zrh}, 2);

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
