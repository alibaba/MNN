//
//  OnnxLSTMMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OnnxExtraManager.hpp"
#include "MNN_generated.h"


namespace MNN {
namespace Express {

class OnnxLSTMTransform : public OnnxExtraManager::Transform{
public:
    virtual EXPRP onExecute(EXPRP expr) const override{
        auto inputs = expr->inputs();
        MNN_CHECK(inputs.size() == 6, "ONNX LSTM should have 6 inputs!");
        auto output = expr->outputs();
        for (auto v : output) {
            auto inputExpr = v.lock();
            if (inputExpr == nullptr) {
                continue;
            }
            for (auto input : inputExpr->inputs()) {
                auto inputExpr = input->expr();
                if (inputExpr.first == expr) {
                    if (1 <= inputExpr.second) {
                        MNN_ERROR("Don't support %s LSTM's multi output\n", expr->name().c_str());
                        return nullptr;
                    }
                }
            }
        }
        std::unique_ptr<OpT> lstm(new OpT);
        lstm->name = expr->name();
        lstm->type = OpType_LSTM;
        lstm->main.type = OpParameter_LSTM;
        
        std::unique_ptr<LSTMT> lstmParam(new LSTMT);
        auto weightI = inputs[1];
        auto weightH = inputs[2];
        auto bias = inputs[3];
        
        auto processWeight = [](VARP& data, BlobT* dst){
            auto weightInfo = data->getInfo();
            const int weightSize = weightInfo->size;
            const auto weightPtr = data->readMap<float>();
            MNN_CHECK(weightPtr != nullptr, "LSTM should have constant weight or bias!");
            const int dimSize = weightInfo->dim.size();
            dst->dims.resize(dimSize);
            memcpy(dst->dims.data(), weightInfo->dim.data(), dimSize * sizeof(int));
            dst->float32s.resize(weightSize);
            memcpy(dst->float32s.data(), weightPtr, weightSize * sizeof(float));
        };
        
        // from IOFG to IFOG for weight
        // from IOFGIOFG to IFOGIFOG for bias
        auto formatChange = [](float* src, bool isBias, int hiddenSize, int inputLength){
            int copyLength = hiddenSize * inputLength;
            if(isBias){
                copyLength = hiddenSize;
            }
            auto tempBuffer = new float[copyLength];
            // copy original O to temp buffer
            memcpy(tempBuffer, src + copyLength, sizeof(float) * copyLength);
            // copy original F to the second stub
            memcpy(src + copyLength, src + 2 * copyLength, sizeof(float) * copyLength);
            // copy temp buffer to the third stub
            memcpy(src + 2 * copyLength, tempBuffer, sizeof(float) * copyLength);
            delete[] tempBuffer;
        };
        
        lstmParam->weightI.reset(new BlobT);
        lstmParam->weightH.reset(new BlobT);
        lstmParam->bias.reset(new BlobT);
        processWeight(weightI, lstmParam->weightI.get());
        processWeight(weightH, lstmParam->weightH.get());
        processWeight(bias, lstmParam->bias.get());
        
        const int weightDim = lstmParam->weightH->dims.size();
        MNN_CHECK(weightDim == 3, "LSTM weight should be 3 dimensions");
        const int hiddenSize = lstmParam->weightH->dims[2];
        const int inputCodeLength = lstmParam->weightI->dims[2];
        
        formatChange(lstmParam->weightI->float32s.data(), false, hiddenSize, inputCodeLength);
        formatChange(lstmParam->weightH->float32s.data(), false, hiddenSize, hiddenSize);
        formatChange(lstmParam->bias->float32s.data(), true, hiddenSize, hiddenSize);
        formatChange(lstmParam->bias->float32s.data() + 4 * hiddenSize, true, hiddenSize, hiddenSize);
        
        lstmParam->outputCount = hiddenSize;
        lstm->main.value = lstmParam.release();
        
        
        auto input0 = inputs[0];
        auto input0Expr = input0->expr().first;
        const auto input0Op = input0Expr->get();
        
        // delete Squeeze before LSTM(there is Squeeze op when nn.LSTM num_layers > 1 and bidirectional is fasle)
        // more details go to https://github.com/pytorch/pytorch/blob/72b9bda9e51fad91f5768d898fd043f8a26dfe99/torch/onnx/symbolic_opset9.py#L1507
        
        // check batch_first according to whether the op type before LSTM is Transpose, because MNN now only support batch_first LSTM, more detailes go to https://github.com/pytorch/pytorch/blob/72b9bda9e51fad91f5768d898fd043f8a26dfe99/torch/onnx/symbolic_opset9.py#L1403
        // delete the Transpose op, because the input of MNN LSTM accept [batch, seq_length, hidden_size],
        // so no need to Transpose the tensor
        
        MNN_CHECK(input0Op->type() == OpType_Permute || input0Op->type() == OpType_Squeeze, "TODO ==> support biLSTM!");
        MNN_CHECK(input0Expr->inputs().size() == 1, "ONNX Transpose|Squeeze should have one input!");
        auto lstmTrueInput = input0Expr->inputs()[0];
        
        if(input0Op->type() == OpType_Permute){
            // create UnSqueeze op, because the input tensor of MNN-LSTM is like [batch, seq_len, 1, hidden_size]
            // but the true lstm should accept 3 dimension tensor
            lstmTrueInput = _Unsqueeze(lstmTrueInput, {2});
        }
        auto originLSTM = Expr::create(lstm.get(), {_Convert(lstmTrueInput, NC4HW4)});
        originLSTM->setName(expr->name());
        auto lstmVar = Variable::create(originLSTM);
        auto res = _Convert(lstmVar, NCHW);
        return res->expr().first;
    }
};

static auto gRegister = [](){
    OnnxExtraManager::get()->insert("LSTM", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLSTMTransform));
    return true;
}();

}
}
