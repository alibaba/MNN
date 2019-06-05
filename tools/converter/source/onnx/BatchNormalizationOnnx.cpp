//
//  BatchNormalizationOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(BatchNormalizationOnnx);

MNN::OpType BatchNormalizationOnnx::opType() {
    return MNN::OpType_BatchNorm;
}
MNN::OpParameter BatchNormalizationOnnx::type() {
    return MNN::OpParameter_BatchNorm;
}

void BatchNormalizationOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                                 std::vector<const onnx::TensorProto*> initializers) {
    DCHECK(initializers.size() == 4) << "BatchNorm Input ERROR";
    auto batchnorm = new MNN::BatchNormT;

    int channels  = 1;
    float epsilon = 0.001;

    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "epsilon") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_FLOAT) << "Node Attribute ERROR";
            epsilon = attributeProto.f();
        }
    }

    const auto gamma = initializers[0];
    const auto beta  = initializers[1];
    const auto mean  = initializers[2];
    const auto var   = initializers[3];
    DCHECK(var->dims_size() == 1) << "Node Attribute ERROR";
    channels = var->dims(0);
    batchnorm->slopeData.resize(channels);
    batchnorm->biasData.resize(channels);
    batchnorm->meanData.resize(channels);
    batchnorm->varData.resize(channels);
    batchnorm->channels = channels;

    if (gamma->float_data_size() != 0) {
        for (int i = 0; i < channels; ++i) {
            batchnorm->slopeData[i] = gamma->float_data(i);
            batchnorm->biasData[i]  = beta->float_data(i);
            batchnorm->meanData[i]  = mean->float_data(i);
            batchnorm->varData[i]   = var->float_data(i) + epsilon;
        }
    } else if (gamma->raw_data().data()) {
        const int copyLen = sizeof(float) * channels;
        ::memcpy(batchnorm->slopeData.data(), gamma->raw_data().data(), copyLen);
        ::memcpy(batchnorm->biasData.data(), beta->raw_data().data(), copyLen);
        ::memcpy(batchnorm->meanData.data(), mean->raw_data().data(), copyLen);
        const float* varPtr = reinterpret_cast<const float*>(var->raw_data().data());
        for (int i = 0; i < channels; ++i) {
            batchnorm->varData[i] = varPtr[i] + epsilon;
        }
    } else {
        DLOG(FATAL) << "BatchNormalization param ERROR!";
    }

    dstOp->main.value = batchnorm;
}

REGISTER_CONVERTER(BatchNormalizationOnnx, BatchNormalization);
