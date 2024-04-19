//
//  SaveHalfFloat.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include "half.hpp"
#include <math.h>

void CastParamsToHalf(std::unique_ptr<MNN::OpT>& op) {
    const auto opType = op->type;
    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_ConvolutionDepthwise:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_DeconvolutionDepthwise:
        {
            auto param           = op->main.AsConvolution2D();
            if (param->quanParameter != nullptr) {
                break;
            }
            const int weightSize = param->weight.size();
            // const int biasSize = param->bias.size();
            std::vector<half_float::half> quantizedFp16Weight;
            quantizedFp16Weight.resize(weightSize);
            std::transform(param->weight.begin(), param->weight.end(), quantizedFp16Weight.begin(),
                            [](float w) {
                w = fmaxf(w, -65504.0f);
                w = fminf(w, 65504.0f);
                return half_float::half(w);
            });
            // std::vector<half_float::half> quantizedFp16Bias;
            // quantizedFp16Bias.resize(biasSize);
            // std::transform(param->bias.begin(), param->bias.end(), quantizedFp16Bias.begin(), [](float
            // b){return half_float::half(b); });
            param->weight.clear();
            // param->bias.clear();

            param->quanParameter.reset(new MNN::IDSTQuanT);
            param->quanParameter->type = 3;
            int8_t* halfWeight         = reinterpret_cast<int8_t*>(quantizedFp16Weight.data());
            param->quanParameter->buffer.assign(halfWeight, halfWeight + sizeof(half_float::half) * weightSize);
            break;
        }
        case MNN::OpType_Const: {
            auto blob = op->main.AsBlob();
            if (blob->dataType == MNN::DataType_DT_FLOAT) {
                blob->dataType = MNN::DataType_DT_HALF;
                blob->uint8s.resize(sizeof(half_float::half) * blob->float32s.size());
                auto size = blob->float32s.size();
                auto dst = (half_float::half*)blob->uint8s.data();
                for (int i=0; i<size; ++i) {
                    float v = blob->float32s[i];
                    v = fmaxf(v, -65504.0f);
                    v = fminf(v, 65504.0f);
                    dst[i] = v;
                }
                blob->float32s.clear();
            }
            break;
        }
        default:
            break;
    }
};
