//
//  OnnxLSTMMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <functional>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "OnnxRNNHelper.hpp"
namespace MNN {
namespace Express {

class OnnxLSTMTransform : public OnnxExtraManager::Transform {
public:
    enum ActivationType {
        Tanh = 0,
        Sigmoid = 1,
        Relu = 2,
    };
    static int _turnStringToAct(std::string actname) {
        if (actname == "Sigmoid") {
            return ActivationType::Sigmoid;
        }
        if (actname == "Tanh") {
            return ActivationType::Tanh;
        }
        if (actname == "Relu") {
            return ActivationType::Relu;
        }
        MNN_PRINT("MNN LSTM Don't support activation: %s\n", actname.c_str());
        return -1;
    }
    static std::function<VARP(VARP)> _selectAct(int act) {
        switch (act) {
            case ActivationType::Tanh:
                return _Tanh;
            case ActivationType::Sigmoid:
                return _Sigmoid;
            case ActivationType::Relu:
                return [](VARP x) {
                    return _Relu(x);
                };
            default:
                break;
        }
        return nullptr;
    }
    // O, Cell
    static std::pair<VARP, VARP> _splitAndAct(VARP Gate, VARP Cell_Init, int hiddenSize, int act0, int act1, int act2) {
        auto splits = _Split(Gate, {4}, 1);
        std::function<VARP(VARP)> act0Function = _selectAct(act0);
        std::function<VARP(VARP)> act1Function = _selectAct(act1);
        std::function<VARP(VARP)> act2Function = _selectAct(act2);
        auto I = act0Function(splits[0]);
        auto O = act0Function(splits[1]);
        auto F = act0Function(splits[2]);
        auto C = act1Function(splits[3]);
        
        auto Cell = I * C + F * _Reshape(Cell_Init, {-1, hiddenSize, 1, 1});
        I = act2Function(Cell);
        O = I * O;
        O = _Reshape(O, {1, -1, hiddenSize});
        Cell = _Reshape(Cell, {1, -1, hiddenSize});
        return std::make_pair(O, Cell);
    }
    static EXPRP _LSTMToWhile(const OpT* lstmOp, std::vector<VARP> inputs, int act0, int act1, int act2) {
        /** Use While and insert Convolution to compute LSTM, then we can quant the weight in LSTM*/
        auto X_Input      = inputs[0];
        auto W            = inputs[1];
        auto R            = inputs[2];
        auto B            = inputs[3];
        VARP O_InitOrigin    = nullptr;
        VARP Cell_InitOrigin = nullptr;
        if (inputs.size() >= 6) {
            O_InitOrigin = inputs[5];
        }
        if (inputs.size() >= 7) {
            Cell_InitOrigin = inputs[6];
        }
        auto wInfo = W->getInfo();
        int direction = wInfo->dim[0];
        auto bInfo = B->getInfo();
        auto rInfo = R->getInfo();
        int hiddenSize = rInfo->dim[2];
        int inputSize = wInfo->dim[2];
        std::vector<VARP> O_InitGroup;
        std::vector<VARP> Cell_InitGroup;
        VARP zeroInit;
        if (nullptr == O_InitOrigin) {
            if (nullptr == zeroInit) {
                zeroInit = _Const(0.0f, std::vector<int>{1, 1, hiddenSize}, NCHW);
            }
            for (int i=0; i<direction; ++i) {
                O_InitGroup.emplace_back(zeroInit);
            }
        } else {
            if (1 == direction) {
                O_InitGroup = {O_InitOrigin};
            } else {
                O_InitGroup = _Split(O_InitOrigin, {direction}, 0);
            }
        }
        if (nullptr == Cell_InitOrigin) {
            if (nullptr == zeroInit) {
                zeroInit = _Const(0.0f, std::vector<int>{1, 1, hiddenSize}, NCHW);
            }
            for (int i=0; i<direction; ++i) {
                Cell_InitGroup.emplace_back(zeroInit);
            }
        } else {
            if (1 == direction) {
                Cell_InitGroup = {Cell_InitOrigin};
            } else {
                Cell_InitGroup = _Split(Cell_InitOrigin, {direction}, 0);
            }
        }
        auto zero = _Unsqueeze(_Scalar<int32_t>(0), {0});
        auto one = _Unsqueeze(_Scalar<int32_t>(1), {0});
        auto negone = _Unsqueeze(_Scalar<int32_t>(-1), {0});
        auto componentVar = _Unsqueeze(_Scalar<int32_t>(4), {0});
        std::vector<VARP> Output;
        std::vector<VARP> OLast;
        std::vector<VARP> CellLast;
        for (int i=0; i<direction; ++i) {
            // FirstPart: Gate = MatMul(X, W, B) :  N * hiddenSize, seqLength * batchSize
            // Gate = Conv(Reshape(X, {seqLength * batch, inputSize, 1, 1}))
            // Gate: seqLength * batch, N * hiddenSize, 1, 1
            VARP FullGate = _makeConvForW(W, B, X_Input, inputSize, i);
            // Make SubGraph
            auto bodyGraphName = lstmOp->name + "_main" + std::to_string(i);
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
                VARP I;
                VARP C;
                VARP F;
                VARP O = _Input({1, -1, hiddenSize}, NCHW);
                O->setName("O");
                auto OI = O;
                VARP Cell = _Input({1, -1, hiddenSize}, NCHW);
                Cell->setName("Cell");
                auto CellI = Cell;
                VARP HR = _makeConvForRStep(O, R, hiddenSize, i, nullptr);
                
                Gate = Gate + HR;
                auto ocell = _splitAndAct(Gate, Cell, hiddenSize, act0, act1, act2);
                O = ocell.first;
                Cell = ocell.second;
                O->setName("O_next");
                Cell->setName("Cell_next");
                auto cond = _Input({}, NCHW, halide_type_of<int>());
                cond->setName("cond");
                std::unique_ptr<OpT> copyOp(new OpT);
                copyOp->type = OpType_Identity;
                EXPRP copyExpr = Expr::create(copyOp.get(), {O}, 1);
                auto OCopy = Variable::create(copyExpr);
                OCopy->setName("O_next_copy");

                auto outputCond = _Scalar<float>(1.0f);
                outputCond->setName("output_cond");
                ExecutorScope::Current()->registerSubGraph(bodyGraphName, {outputCond, inputShape, GateFull, O, Cell, OCopy}, {step, cond, inputShape, GateFull, OI, CellI});
            }
            auto inputShape = _Shape(inputs[0], true);
            auto seqLengthVar = _Slice(inputShape, _Unsqueeze(_Scalar<int32_t>(0), {0}), one);

            // Make Copy Op to fuse three varps
            std::unique_ptr<OpT> loopOp(new OpT);
            loopOp->type = OpType_While;
            loopOp->main.value = new WhileParamT;
            loopOp->main.type = OpParameter_WhileParam;
            auto whileP = loopOp->main.AsWhileParam();
            whileP->body_graph = bodyGraphName;
            auto cond = _Scalar<int>(1);
            auto whileInputs = std::vector<VARP>{seqLengthVar, cond, inputShape, FullGate, O_InitGroup[i], Cell_InitGroup[i]};
            auto whileExpr = Expr::create(loopOp.get(), whileInputs, 5);
            auto directionO = Variable::create(whileExpr, 4);
            if (1 == i) {
                directionO = _Reverse(directionO, _Scalar<int>(0));
            }
            Output.emplace_back(directionO);
            OLast.emplace_back(Variable::create(whileExpr, 2));
            CellLast.emplace_back(Variable::create(whileExpr, 3));
        }

        std::unique_ptr<OpT> copyOp(new OpT);
        copyOp->type = OpType_Identity;
        EXPRP resultExpr;
        if (1 == direction) {
            resultExpr = Expr::create(copyOp.get(), {Output[0], OLast[0], CellLast[0]}, 3);
        } else {
            auto o0 = _Concat(Output, 1);
            auto o1 = _Concat(OLast, 0);
            auto o2 = _Concat(CellLast, 0);
            resultExpr = Expr::create(copyOp.get(), {o0, o1, o2}, 3);
        }
        resultExpr->setName(lstmOp->name);
        return resultExpr;
    }
    static EXPRP singleLSTMOpt(const OpT* lstmOp, std::vector<VARP> inputs, int act0, int act1, int act2) {
        auto X_Input      = inputs[0];
        auto W            = inputs[1];
        auto R            = inputs[2];
        auto B            = inputs[3];
        VARP O_Init    = inputs[5];
        VARP Cell_Init = inputs[6];
        auto wInfo = W->getInfo();
        auto bInfo = B->getInfo();
        auto rInfo = R->getInfo();
        auto XInfo = X_Input->getInfo();
        int batchSize = XInfo->dim[1];
        int hiddenSize = rInfo->dim[2];
        int inputSize = wInfo->dim[2];
        VARP Gate = _makeConvForW(W, B, X_Input, inputSize, 0);
        VARP HR = _makeConvForRStep(O_Init, R, hiddenSize, 0, nullptr);
        Gate = Gate + HR;
        auto ocell = _splitAndAct(Gate, Cell_Init, hiddenSize, act0, act1, act2);
        auto O = ocell.first;
        auto Cell = ocell.second;
        // Make Copy Op to fuse three varps
        std::unique_ptr<OpT> copyOp(new OpT);
        copyOp->type = OpType_Identity;
        
        auto fuseOutput = _Unsqueeze(O, {0});
        auto resultExpr = Expr::create(copyOp.get(), {fuseOutput, O, Cell}, 3);
        return resultExpr;
    }
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        if (inputs.size() == 8) {
            MNN_ERROR("MNN LSTM not support 8th input (peepholes)\n");
            return nullptr;
        }
        if (inputs.size() >= 5 && inputs[4].get() != nullptr) {
            MNN_ERROR("MNN LSTM not support sequence_lens, all batch must be seq_length, the result may has error\n");
            // return nullptr;
        }
        std::unique_ptr<OpT> lstm(new OpT);
        lstm->name       = expr->name();
        if (expr->get()->main_as_Extra()->type()->str() == "RNN") {
            lstm->type = OpType_RNN;
        } else {
            lstm->type = OpType_LSTM;
        }
        lstm->main.type  = OpParameter_LSTM;
        lstm->main.value = new LSTMT;
        int act0 = Sigmoid;
        int act1 = Tanh;
        int act2 = Tanh;
        {
            auto extra = expr->get()->main_as_Extra();
            auto attr  = extra->attr();
            if (nullptr != attr) {
                for (int i = 0; i < attr->size(); ++i) {
                    auto attUnit = attr->GetAs<Attribute>(i);
                    if (attUnit->key()->str() == "hidden_size") {
                        lstm->main.AsLSTM()->outputCount = attUnit->i();
                        continue;
                    }
                    if (attUnit->key()->str() == "activations") {
                        auto s = attUnit->list();
                        if (nullptr != s && nullptr != s->s() && 3 <= s->s()->size()) {
                            act0 = _turnStringToAct(s->s()->GetAsString(0)->str());
                            act1 = _turnStringToAct(s->s()->GetAsString(1)->str());
                            act2 = _turnStringToAct(s->s()->GetAsString(2)->str());
                        } else {
                            MNN_ERROR("Load activations error for %s\n", expr->name().c_str());
                        }
                        continue;
                    }
                }
            }
        }
        if (act0 < 0 || act1 < 0 || act2 < 0) {
            return nullptr;
        }
        if (inputs.size() < 4 || inputs[3].get() == nullptr) {
            // Bias is zero
            auto shapeWeight = _Shape(inputs[1], NCHW);
            auto shapeBias = _Split(shapeWeight, {2, 1})[0];
            float v = 0.0f;
            auto zeroScalar = _Const(&v, {}, NCHW, halide_type_of<float>());
            auto biasWR = _Fill(shapeBias, zeroScalar);
            if (inputs.size() < 4) {
                inputs.emplace_back(biasWR);
            } else {
                inputs[3] = biasWR;
            }
        } else {
            // onnx docs guarantee bias shape is [num_direction, 8 * hidden_size], we split it to 2x [num_dicection, 4 * hidden_size] (W/R), then add together
            auto biasWR = _Split(inputs[3], {2}, 1);
            inputs[3] = _Add(biasWR[0], biasWR[1]);
        }
        auto inputInfo = inputs[0]->getInfo();
        auto weightInfo = inputs[1]->getInfo();
        if (nullptr != inputInfo && nullptr != weightInfo && inputInfo->dim.size() > 0 && weightInfo->dim.size() > 0) {
            if (inputInfo->dim[0] == 1 && lstm->type == OpType_LSTM && weightInfo->dim[0] == 1 && inputs.size() >= 7) {
                // SeqLength = 1, use unroll lstm
                inputs[3].fix(VARP::CONSTANT);
                if (inputs[2]->readMap<float>() != nullptr && inputs[3]->readMap<float>() != nullptr && inputs[1]->readMap<float>() != nullptr) {
                    auto lstmExpr = singleLSTMOpt(lstm.get(), inputs, act0, act1, act2);
                    lstmExpr->setName(expr->name());
                    for (int i = 0; i < lstmExpr->outputSize(); ++i) {
                        Variable::create(lstmExpr, i)->setName(expr->outputName(i));
                    }
                    return lstmExpr;
                }
            }
        }
        auto config = Global<modelConfig>::Get();
        lstm->name = expr->name();
        if (!config->useOriginRNNImpl) {
            if (nullptr != weightInfo && weightInfo->dim.size() > 0) {
                if (lstm->type == OpType_LSTM) {
                    inputs[3].fix(VARP::CONSTANT);
                    if (inputs[2]->readMap<float>() != nullptr && inputs[3]->readMap<float>() != nullptr && inputs[1]->readMap<float>() != nullptr) {
                        MNN_PRINT("Use While to compute LSTM, if don't want it, add --useOriginRNNImpl \n");
                        auto lstmExpr = _LSTMToWhile(lstm.get(), inputs, act0, act1, act2);
                        lstmExpr->setName(expr->name());
                        for (int i = 0; i < lstmExpr->outputSize(); ++i) {
                            Variable::create(lstmExpr, i)->setName(expr->outputName(i));
                        }
                        return lstmExpr;
                    }
                }
            }
        }
        if (inputs.size() >= 5) {
            inputs.erase(inputs.begin() + 4); // ignore sequence_lens
        }
        // Y, Y_h, Y_c
        auto originLSTM = Expr::create(lstm.get(), inputs, (lstm->type == OpType_RNN ? 2 : 3));
        originLSTM->setName(expr->name());
        for (int i = 0; i < expr->outputSize(); ++i) {
            Variable::create(originLSTM, i)->setName(expr->outputName(i));
        }
        return originLSTM;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("LSTM", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLSTMTransform));
    OnnxExtraManager::get()->insert("RNN", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLSTMTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
