//
//  LayerNormOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(LayerNormOnnx);

MNN::OpType LayerNormOnnx::opType() {
    return MNN::OpType_LayerNorm;
}
MNN::OpParameter LayerNormOnnx::type() {
    return MNN::OpParameter_LayerNorm;
}

void LayerNormOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
    std::unique_ptr<MNN::LayerNormT> param(new MNN::LayerNormT);

    float slope = 1e-10;
    const auto attrSize = onnxNode->attribute_size();
    std::vector<int32_t> axis_var_;
    for (int i = 0; i < attrSize; ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();

        if (attributeName == "axis") {
            axis_var_.push_back(attributeProto.i());
        } else if (attributeName == "epsilon") {
            param->epsilon = attributeProto.f();
        } else if (attributeName == "group") {
            param->group = attributeProto.i();
        } else {
            DLOG(ERROR) << "TODO!";
        }
    }
    param->axis = axis_var_;

    std::vector<const onnx::TensorProto*> initializers;
    for (int k = 0; k < onnxNode->input_size(); ++k) {
        const auto& inputName = onnxNode->input(k);
        const auto it         = scope->mInitializers.find(inputName);
        if (it != scope->mInitializers.end()) {
            initializers.push_back(it->second);
        }
    }
    const int size = initializers.size();
    DCHECK(size <= 2 && size >= 1) << "Gemm Input ERROR!";

    const auto gammaProto = initializers[0];
    DCHECK(1 == gammaProto->dims_size()) << "LayerNorm Gamma dimensions should be 1";

    int gammaSize = 1;
    for (int i = 0; i < gammaProto->dims_size(); ++i) {
        gammaSize *= gammaProto->dims(i);
    }

    std::vector<float> gammaContainer(gammaSize);
    auto gammaPtr = gammaContainer.data();

    if (gammaProto->float_data_size() != 0) {
        for (int i = 0; i < gammaSize; ++i) {
            gammaPtr[i] = gammaProto->float_data(i);
        }
    } else if (gammaProto->raw_data().data()) {
        ::memcpy(gammaPtr, reinterpret_cast<const float*>(gammaProto->raw_data().data()), gammaSize * sizeof(float));
    } else {
        DLOG(ERROR) << "ERROR";
    }


    const auto betaProto = size == 2 ? initializers[1] : nullptr;
    std::vector<float> betaContainer;
    if(betaProto) {
        DCHECK(1 == betaProto->dims_size()) << "LayerNorm Beta dimensions should be 1";

        int betaSize = 1;
        for (int i = 0; i < betaProto->dims_size(); ++i) {
            betaSize *= betaProto->dims(i);
        }
        betaContainer.resize(betaSize);
        auto betaPtr = betaContainer.data();

        if (betaProto->float_data_size() != 0) {
            for (int i = 0; i < betaSize; ++i) {
                betaPtr[i] = betaProto->float_data(i);
            }
        } else if (betaProto->raw_data().data()) {
            ::memcpy(betaPtr, reinterpret_cast<const float*>(betaProto->raw_data().data()), betaSize * sizeof(float));
        } else {
            DLOG(ERROR) << "ERROR";
        }
    }

    param->gamma = gammaContainer;
    param->beta  = betaContainer;

    dstOp->main.value = param.release();
    return;
}

REGISTER_CONVERTER(LayerNormOnnx, LayerNorm);
