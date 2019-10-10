//
//  ConvertBinaryToElementwise.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
using namespace MNN;
class ConvertBinaryToElementwise : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& mNet = net;
        for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end(); iter++) {
            auto op = iter->get();

            if (op->type != MNN::OpType_BinaryOp) {
                continue;
            }

            auto param = op->main.AsBinaryOp();
            if (param->opType != BinaryOpOperation_MUL && param->opType != BinaryOpOperation_ADD &&
                param->opType != BinaryOpOperation_SUB) {
                continue;
            }
            const int inputNum = op->inputIndexes.size();
            DCHECK(inputNum == 2) << "BinaryOp should have two inputs";

            const int inputIndex0 = op->inputIndexes[0];
            auto inputOp0         = PostTreatUtils::_findOpByOutputIndex(inputIndex0, mNet.get());
            const int inputIndex1 = op->inputIndexes[1];
            auto inputOp1         = PostTreatUtils::_findOpByOutputIndex(inputIndex1, mNet.get());
            bool readyToChange = (inputOp0->type == MNN::OpType_Convolution || inputOp0->type == MNN::OpType_Eltwise) &&
                                 (inputOp1->type == MNN::OpType_Convolution || inputOp1->type == MNN::OpType_Eltwise);

            if (readyToChange) {
                // convert binary op to elementwise op
                auto elementParam = new MNN::EltwiseT;
                switch (param->opType) {
                    case BinaryOpOperation_MUL:
                        elementParam->type = EltwiseType_PROD;
                        break;
                    case BinaryOpOperation_ADD:
                        elementParam->type = EltwiseType_SUM;
                        break;
                    case BinaryOpOperation_SUB:
                        elementParam->type = EltwiseType_SUB;
                        break;
                    default:
                        break;
                }
                op->type = MNN::OpType_Eltwise;
                op->main.Reset();
                op->main.type  = OpParameter_Eltwise;
                op->main.value = elementParam;
            }
        }
        return true;
    }
};
static PostConverterRegister<ConvertBinaryToElementwise> __l("ConvertBinaryToElementwise");
