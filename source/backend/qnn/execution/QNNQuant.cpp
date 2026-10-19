//
//  QNNQuant.cpp
//  MNN
//
//  Created by MNN on b'2025/05/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNQuant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNQuant::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mBackend->requiresQuantizedGraph()) {
        // The V66 DSP graph stays fixed point end to end. MNN's synthetic
        // FloatToInt8 boundary therefore becomes either an identity reshape or
        // a fixed-point Convert, never a float Cast. The DSP Quantize op only
        // accepts FLOAT_32 input; Convert is the supported SFixed8-to-SFixed8
        // requantization primitive.
        const auto* input = mBackend->getNativeTensor(inputs[0]);
        const auto* output = mBackend->getNativeTensor(outputs[0]);
        if (input->v1.dataType == QNN_DATATYPE_FLOAT_32) {
            // Constants are not covered by the end-to-end activation
            // fixed-point registration in QnnBackend. Although host-side
            // validation accepts Quantize here, the V66 skeleton rejects it
            // during graph finalization. Fold this constant conversion while
            // building the graph and expose it as a native SFixed8 tensor.
            const float scale =
                output->v1.quantizeParams.scaleOffsetEncoding.scale;
            const int32_t offset =
                output->v1.quantizeParams.scaleOffsetEncoding.offset;
            if (!(scale > 0.0f)) {
                return INVALID_VALUE;
            }
            std::vector<int8_t> quantized(inputs[0]->elementSize());
            const float* source = inputs[0]->host<float>();
            for (size_t index = 0; index < quantized.size(); ++index) {
                const int value =
                    static_cast<int>(std::round(source[index] / scale)) -
                    offset;
                quantized[index] = static_cast<int8_t>(
                    std::max(-128, std::min(127, value)));
            }
            std::vector<uint32_t> dimensions(
                output->v1.dimensions,
                output->v1.dimensions + output->v1.rank);
            const auto folded = this->createStaticTensor(
                "dsp_folded_constant", QNN_DATATYPE_SFIXED_POINT_8,
                dimensions, quantized.data(), output->v1.quantizeParams);
            mNodeType =
                mBackend->isDspBackend() ? "Reshape" : "Convert";
            mInputs.push_back(*(folded->getNativeTensor()));
            mOutputs.push_back(*output);
            mBackend->addNodeToGraph(
                mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(),
                mNodeType.c_str(), mParams, mInputs, mOutputs);
            return NO_ERROR;
        }
        const float inputScale =
            input->v1.quantizeParams.scaleOffsetEncoding.scale;
        const float outputScale =
            output->v1.quantizeParams.scaleOffsetEncoding.scale;
        mNodeType =
            mBackend->isDspBackend() &&
                    std::fabs(inputScale - outputScale) <=
                        std::max(inputScale, outputScale) * 1.0e-6f
                ? "Reshape"
                : "Convert";
        mInputs.push_back(*input);
        mOutputs.push_back(*output);
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(),
                                 mPackageName.c_str(), mNodeType.c_str(),
                                 mParams, mInputs, mOutputs);
        return NO_ERROR;
    }
    this->createStageTensor("Cast", QNN_DATATYPE_FLOAT_32, getNHWCShape(outputs[0]));
     // Stage one  fp16 -> fp32
    {
        mNodeType = "Cast";
        std::string name = mNodeName + "_Cast";
    
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input
        mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage tensor
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Stage two  fp32 -> int8
    {
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Quantize";
        std::string name = mNodeName;
    
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage tensor
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    return NO_ERROR;
}


class QNNQuantCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNQuant(backend, op);
    }
};

ErrorCode QNNDeQuant::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mBackend->requiresQuantizedGraph()) {
        const auto* input = mBackend->getNativeTensor(inputs[0]);
        const auto* output = mBackend->getNativeTensor(outputs[0]);
        const float inputScale =
            input->v1.quantizeParams.scaleOffsetEncoding.scale;
        const float outputScale =
            output->v1.quantizeParams.scaleOffsetEncoding.scale;
        // See the FloatToInt8 branch above: DSP Quantize cannot consume an
        // already-fixed-point tensor. Convert preserves the intended change
        // in scale/offset without introducing an FP32 island.
        mNodeType =
            mBackend->isDspBackend() &&
                    std::fabs(inputScale - outputScale) <=
                        std::max(inputScale, outputScale) * 1.0e-6f
                ? "Reshape"
                : "Convert";
        mInputs.push_back(*input);
        mOutputs.push_back(*output);
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(),
                                 mPackageName.c_str(), mNodeType.c_str(),
                                 mParams, mInputs, mOutputs);
        return NO_ERROR;
    }
     // Stage one  int8 -> fp16
    {
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Dequantize";
        std::string name = mNodeName;
    
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    return NO_ERROR;
}


class QNNDeQuantCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNDeQuant(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNQuantCreator, OpType_FloatToInt8)
REGISTER_QNN_OP_CREATOR(QNNDeQuantCreator, OpType_Int8ToFloat)
#endif
} // end namespace QNN
} // end namespace MNN
