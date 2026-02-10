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
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNUnary::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    if (UnaryOpOperation_SILU == mOp->main_as_UnaryOp()->opType()) {
        Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
        this->createStageTensor("Stage", dataType, getNHWCShape(inputs[0]), outputs[0]);
        auto input = inputs[0];
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(input)));
            mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
            std::string name = mNodeName + "Sigmoid__";
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), "Sigmoid", mParams, mInputs, mOutputs);
        }
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(input)));
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
            std::string name = mNodeName + "ElementWiseMultiply__";
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), "ElementWiseMultiply", mParams, mInputs, mOutputs);
        }

        return NO_ERROR;
    }

    // UnaryOpOperation_SQUARE.
    if(UnaryOpOperation_SQUARE == mOp->main_as_UnaryOp()->opType())
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "ElementWiseMultiply";
        for (int i = 0; i < mParamTensorWrappers.size(); i++) {
            mParams.push_back(*(mParamTensorWrappers[i]->getNativeParam()));
        }

        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input0
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input0
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
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
                                    std::set<UnaryOpOperation> supportedUnaryTypes = {
                                        UnaryOpOperation_ABS,
                                        UnaryOpOperation_EXP,
                                        UnaryOpOperation_SQRT,
                                        UnaryOpOperation_RSQRT,
                                        UnaryOpOperation_LOG,
                                        UnaryOpOperation_SIN,
                                        UnaryOpOperation_ASIN,
                                        UnaryOpOperation_COS,
                                        UnaryOpOperation_TANH,
                                        UnaryOpOperation_FLOOR,
                                        UnaryOpOperation_SIGN,
                                        UnaryOpOperation_SIGMOID,
                                        UnaryOpOperation_SQUARE,
                                        UnaryOpOperation_NEG,
                                        UnaryOpOperation_HARDSWISH,
                                        UnaryOpOperation_GELU,
                                        UnaryOpOperation_GELU_STANDARD,
                                        UnaryOpOperation_SILU
                                    };
        auto opType = op->main_as_UnaryOp()->opType();
        if (supportedUnaryTypes.find(opType) == supportedUnaryTypes.end()) {
            MNN_ERROR("Don't support %d opType in QNNUnary\n", opType);
            return nullptr;
        }
        return new QNNUnary(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNUnaryCreator, OpType_UnaryOp)
#endif
} // end namespace QNN
} // end namespace MNN


