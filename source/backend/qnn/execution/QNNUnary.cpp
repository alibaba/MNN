//
//  QNNUnary.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNUnary.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNUnary::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    if (UnaryOpOperation_SILU == mOp->main_as_UnaryOp()->opType()) {
        Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
        this->createStageTensor("Stage", dataType, getNHWCShape(inputs[0]));
        auto input = inputs[0];
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(input)));
            mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
            std::string name = "Sigmoid__";
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), "Sigmoid", mParams, mInputs, mOutputs);
        }
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(input)));
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
            std::string name = "ElementWiseMultiply__";
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), "ElementWiseMultiply", mParams, mInputs, mOutputs);
        }

        return NO_ERROR;
    }
    std::map<UnaryOpOperation, std::string> unaryMap {
        {UnaryOpOperation_ABS, "ElementWiseAbs"},
        {UnaryOpOperation_EXP, "ElementWiseExp"},
        {UnaryOpOperation_SQRT, "ElementWiseSquareRoot"},
        {UnaryOpOperation_RSQRT, "ElementWiseRsqrt"},
        {UnaryOpOperation_LOG, "ElementWiseLog"},
        {UnaryOpOperation_RECIPROCAL, ""},
        {UnaryOpOperation_SIN, "ElementWiseSin"},
        {UnaryOpOperation_ASIN, "ElementWiseAsin"},
        {UnaryOpOperation_SINH, ""},
        {UnaryOpOperation_ASINH, ""},
        {UnaryOpOperation_COS, "ElementWiseCos"},
        {UnaryOpOperation_ACOS, ""},
        {UnaryOpOperation_COSH, ""},
        {UnaryOpOperation_ACOSH, ""},
        {UnaryOpOperation_TAN, ""},
        {UnaryOpOperation_ATAN, ""},
        {UnaryOpOperation_TANH, "Tanh"},
        {UnaryOpOperation_ATANH, ""},
        {UnaryOpOperation_ERF, ""},
        {UnaryOpOperation_CEIL, ""},
        {UnaryOpOperation_FLOOR, "ElementWiseFloor"},
        {UnaryOpOperation_ROUND, ""},
        {UnaryOpOperation_SIGN, "ElementWiseSign"},
        {UnaryOpOperation_SIGMOID, "Sigmoid"},
        {UnaryOpOperation_LOG1P, ""},
        {UnaryOpOperation_SQUARE, ""},
        {UnaryOpOperation_NEG, "ElementWiseNeg"},
        {UnaryOpOperation_HARDSWISH, "HardSwish"},
        {UnaryOpOperation_GELU, "Gelu"},
        {UnaryOpOperation_GELU_STANDARD, "Gelu"},
        {UnaryOpOperation_EXPM1, ""},
        {UnaryOpOperation_ERFC, ""},
        {UnaryOpOperation_BNLL, ""},
        {UnaryOpOperation_ERFINV, ""},
        {UnaryOpOperation_SILU, ""}
    };
    auto opType = mOp->main_as_UnaryOp()->opType();
    auto iter = unaryMap.find(opType);
    if (iter == unaryMap.end() || iter->second.empty()) {
        MNN_ERROR("Don't support %d opType in QNNUnary\n", opType);
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }
    mNodeType = iter->second;

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


class QNNUnaryCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNUnary(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNUnaryCreator, OpType_UnaryOp)

} // end namespace QNN
} // end namespace MNN


