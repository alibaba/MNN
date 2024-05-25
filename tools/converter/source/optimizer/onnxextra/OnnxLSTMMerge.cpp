//
//  OnnxLSTMMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include <MNN/expr/ExprCreator.hpp>

namespace MNN {
namespace Express {

class OnnxLSTMTransform : public OnnxExtraManager::Transform {
public:
    static EXPRP singleLSTMOpt(const OpT* lstmOp, std::vector<VARP> inputs) {
        auto X_Input      = inputs[0];
        auto W            = inputs[1];
        auto R            = inputs[2];
        auto B            = inputs[3];
        VARP O_Init    = nullptr;
        VARP Cell_Init = nullptr;
        if (inputs.size() >= 5) {
            O_Init = inputs[4];
        }
        if (inputs.size() >= 6) {
            Cell_Init = inputs[5];
        }
        auto wInfo = W->getInfo();
        auto bInfo = B->getInfo();
        auto rInfo = R->getInfo();
        auto XInfo = X_Input->getInfo();
        int batchSize = XInfo->dim[1];
        int hiddenSize = rInfo->dim[2];
        int inputSize = wInfo->dim[2];
        VARP Gate;
        {
            std::unique_ptr<OpT> matmulOp(new OpT);
            matmulOp->type = OpType_Convolution;
            matmulOp->main.value = new Convolution2DT;
            matmulOp->main.type = OpParameter_Convolution2D;
            matmulOp->main.AsConvolution2D()->common.reset(new Convolution2DCommonT);
            matmulOp->main.AsConvolution2D()->common->outputCount = wInfo->dim[1];
            matmulOp->main.AsConvolution2D()->common->inputCount = wInfo->dim[2];
            matmulOp->main.AsConvolution2D()->weight.resize(wInfo->dim[1] * wInfo->dim[2]);
            ::memcpy(matmulOp->main.AsConvolution2D()->weight.data(), W->readMap<float>(), matmulOp->main.AsConvolution2D()->weight.size() * sizeof(float));
            matmulOp->main.AsConvolution2D()->bias.resize(matmulOp->main.AsConvolution2D()->common->outputCount);
            ::memcpy(matmulOp->main.AsConvolution2D()->bias.data(), B->readMap<float>(), matmulOp->main.AsConvolution2D()->bias.size() * sizeof(float));
            auto convX = _Reshape(X_Input, {-1, inputSize, 1, 1});
            Gate = Variable::create(Expr::create(matmulOp.get(), {convX}));
        }
        VARP HR;
        {
            std::unique_ptr<OpT> matmulOp(new OpT);
            matmulOp->type = OpType_Convolution;
            matmulOp->main.value = new Convolution2DT;
            matmulOp->main.type = OpParameter_Convolution2D;
            matmulOp->main.AsConvolution2D()->common.reset(new Convolution2DCommonT);
            matmulOp->main.AsConvolution2D()->common->outputCount = rInfo->dim[1];
            matmulOp->main.AsConvolution2D()->common->inputCount = rInfo->dim[2];
            matmulOp->main.AsConvolution2D()->weight.resize(rInfo->dim[1] * rInfo->dim[2]);
            ::memcpy(matmulOp->main.AsConvolution2D()->weight.data(), R->readMap<float>(), matmulOp->main.AsConvolution2D()->weight.size() * sizeof(float));
            matmulOp->main.AsConvolution2D()->bias.resize(matmulOp->main.AsConvolution2D()->common->outputCount);
            ::memset(matmulOp->main.AsConvolution2D()->bias.data(), 0, matmulOp->main.AsConvolution2D()->bias.size() * sizeof(float));
            auto convX = _Reshape(O_Init, {-1, hiddenSize, 1, 1});
            HR = Variable::create(Expr::create(matmulOp.get(), {convX}));
        }
        
        Gate = Gate + HR;
        auto splits = _Split(Gate, {4}, 1);
        auto I = _Sigmoid(splits[0]);
        auto O = _Sigmoid(splits[1]);
        auto F = _Sigmoid(splits[2]);
        auto C = _Tanh(splits[3]);
        
        auto Cell = I * C + F * _Reshape(Cell_Init, {-1, hiddenSize, 1, 1});
        I = _Tanh(Cell);
        O = I * O;
        O = _Reshape(O, {1, -1, hiddenSize});
        Cell = _Reshape(Cell, {1, -1, hiddenSize});
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
            MNN_ERROR("MNN LSTM not support sequence_lens, all batch must be seq_length\n");
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
        {
            auto extra = expr->get()->main_as_Extra();
            auto attr  = extra->attr();
            if (nullptr != attr) {
                for (int i = 0; i < attr->size(); ++i) {
                    auto attUnit = attr->GetAs<Attribute>(i);
                    if (attUnit->key()->str() == "hidden_size") {
                        lstm->main.AsLSTM()->outputCount = attUnit->i();
                    }
                }
            }
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
        if (inputs.size() >= 5) {
            inputs.erase(inputs.begin() + 4); // ignore sequence_lens
        }
        auto inputInfo = inputs[0]->getInfo();
        auto weightInfo = inputs[1]->getInfo();
        if (nullptr != inputInfo && nullptr != weightInfo) {
            if (inputInfo->dim[0] == 1 && lstm->type == OpType_LSTM && weightInfo->dim[0] == 1 && inputs.size() >= 6) {
                // SeqLength = 1
                inputs[3].fix(VARP::CONSTANT);
                if (inputs[2]->readMap<float>() != nullptr && inputs[3]->readMap<float>() != nullptr && inputs[1]->readMap<float>() != nullptr) {
                    auto lstmExpr = singleLSTMOpt(lstm.get(), inputs);
                    lstmExpr->setName(expr->name());
                    for (int i = 0; i < lstmExpr->outputSize(); ++i) {
                        Variable::create(lstmExpr, i)->setName(expr->outputName(i));
                    }
                    return lstmExpr;
                }
            }
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
