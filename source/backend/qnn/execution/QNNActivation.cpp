//
//  QNNActivation.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNActivation.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNActivation::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto opType = mOp->type();
    switch (opType) {
        case OpType_ReLU: {
            const auto *relu = mOp->main_as_Relu();
            const float slope = relu == nullptr ? 0.0f : relu->slope();
            if (slope == 0.0f) {
                mNodeType = "Relu";
                break;
            }

            // MNN represents LeakyReLU as Relu with a non-zero slope. QNN's
            // Relu op has no slope parameter, so preserve the model semantics
            // with a scalar PRelu alpha tensor in the input's native type.
            mNodeType = "Prelu";
            const Qnn_DataType_t dataType =
                mBackend->isDedicatedQnnSession()
                    ? mBackend->getNativeTensor(inputs[0])->v1.dataType
                    : (mBackend->getUseFP16() ? QNN_DATATYPE_FLOAT_16
                                             : QNN_DATATYPE_FLOAT_32);
            std::shared_ptr<QNNTensorWrapper> alpha;
            if (mBackend->requiresQuantizedGraph() &&
                dataType == QNN_DATATYPE_SFIXED_POINT_8) {
                // QNN DSP requires every fixed-point tensor, including a
                // scalar PReLU slope, to carry an explicit quantization
                // encoding. Do not send the float buffer through the
                // fixed-point tensor path: that also gives it the wrong
                // element width.
                Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS;
                Qnn_ScaleOffset_t scaleOffset = {};
                scaleOffset.scale =
                    std::max(std::abs(slope) / 127.0f, 1.0e-8f);
                scaleOffset.offset = 0;
                quantize.encodingDefinition = QNN_DEFINITION_DEFINED;
                quantize.quantizationEncoding =
                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                quantize.scaleOffsetEncoding = scaleOffset;
                const int8_t quantizedSlope = static_cast<int8_t>(
                    std::round(slope / scaleOffset.scale));
                alpha = this->createStaticTensor(
                    "alpha", dataType, std::vector<uint32_t>({1}),
                    &quantizedSlope, quantize);
            } else {
                alpha = this->createStaticFloatTensor(
                    mBackend->isDedicatedQnnSession() ? "alpha" : "coeff",
                    dataType, std::vector<uint32_t>({1}), &slope);
            }
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
            mInputs.push_back(*(alpha->getNativeTensor()));
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
            mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(),
                                     mPackageName.c_str(), mNodeType.c_str(),
                                     mParams, mInputs, mOutputs);
            return NO_ERROR;
        }
        case OpType_ReLU6:
            mNodeType = "ReluMinMax";
            this->createParamScalar("min_value", mOp->main_as_Relu6()->minValue());
            this->createParamScalar("max_value", mOp->main_as_Relu6()->maxValue());
            break;
        case OpType_Sigmoid:
            mNodeType = "Sigmoid";
            break;
        case OpType_ELU:
            mNodeType = "Elu";
            this->createParamScalar("alpha", mOp->main_as_ELU()->alpha());
            break;
        default:
            MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


class QNNActivationCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNActivation(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNActivationCreator, OpType_ReLU)
REGISTER_QNN_OP_CREATOR(QNNActivationCreator, OpType_ReLU6)
REGISTER_QNN_OP_CREATOR(QNNActivationCreator, OpType_Sigmoid)
REGISTER_QNN_OP_CREATOR(QNNActivationCreator, OpType_ELU)
#endif
} // end namespace QNN
} // end namespace MNN
