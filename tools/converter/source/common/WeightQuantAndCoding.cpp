//
//  WeightQuantAndCoding.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include "cpp/IDSTEncoder.hpp"

static float findAbsMax(const float *weights, const int count) {
    float absMax = fabs(weights[0]);
    for (int i = 1; i < count; i++) {
        float value = fabs(weights[i]);
        if (value > absMax) {
            absMax = value;
        }
    }

    return absMax;
}

static std::vector<float> findMinMax(const float *weights, const int count) {
    float min = weights[0];
    float max = weights[0];

    for (int i = 1; i < count; i++) {
        float value = weights[i];
        if (value > max) {
            max = value;
        }
        if (value < min) {
            min = value;
        }
    }

    return {min, max};
}

void WeightQuantAndCoding(std::unique_ptr<MNN::OpT>& op, const modelConfig& config) {
    const auto opType = op->type;
    // config.weightQuantBits only control weight quantization for float convolution
    // by default, do coding for convint8 and depthwiseconvint8, if there is any
    if ((config.weightQuantBits == 0) && (
        opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8)) {
        return;
    }

    if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise &&
        opType != MNN::OpType_Deconvolution && opType != MNN::OpType_DeconvolutionDepthwise &&
        opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8) {
            return;
    }

    int bits = 8;
    if ((config.weightQuantBits > 0) && (
        opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8)) {
        bits = config.weightQuantBits;
    }
    // Bits must from 2-8
    bits = std::max(bits, 2);
    bits = std::min(bits, 8);

    auto param           = op->main.AsConvolution2D();
    auto& common = param->common;
    if (param->quanParameter.get() != nullptr) {
        return;
    }

    int weightSize = param->weight.size();
    if (opType == MNN::OpType_ConvInt8 || opType == MNN::OpType_DepthwiseConvInt8) {
        weightSize = param->symmetricQuan->weight.size();
    }
    int kernelNum = common->outputCount;
    int kernelSize = weightSize / kernelNum;

    bool asymmetricQuantFlag = config.weightQuantAsymmetric;

    float threshold = (float)(1 << (bits - 1)) - 1.0f;
    float clampMin = -threshold;
    if (asymmetricQuantFlag) {
        clampMin = -threshold - 1;
    }
    std::vector<float> weightData, scales;
    std::vector<int8_t> quantWeights;

    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_ConvolutionDepthwise:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_DeconvolutionDepthwise: {
            weightData = param->weight;

            if (asymmetricQuantFlag) {
                scales.resize(kernelNum*2);
                for (int k = 0; k < kernelNum; k++) {
                    int beginIndex = k * kernelSize;
                    auto minAndMax = findMinMax(weightData.data() + beginIndex, kernelSize);
                    float min = minAndMax[0];
                    float max = minAndMax[1];
                    float scale = (max - min) / (threshold - clampMin);

                    scales[2*k] = min;
                    scales[2*k+1] = scale;

                    for (int ii = 0; ii < kernelSize; ii++) {
                        float* ptr = weightData.data() + beginIndex;
                        int8_t quantValue = int8_t(std::round((ptr[ii] - min) / scale + clampMin));
                        quantWeights.emplace_back(quantValue);
                    }
                }
            } else {
                scales.resize(kernelNum);
                for (int k = 0; k < kernelNum; k++) {
                    int beginIndex = k * kernelSize;
                    auto absMax = findAbsMax(weightData.data() + beginIndex, kernelSize);

                    scales[k] = absMax / threshold;

                    for (int ii = 0; ii < kernelSize; ii++) {
                        float* ptr = weightData.data() + beginIndex;
                        int8_t quantValue = int8_t(std::round(ptr[ii] / scales[k]));
                        quantWeights.emplace_back(quantValue);
                    }
                }
            }

            break;
        }
        case MNN::OpType_ConvInt8:
        case MNN::OpType_DepthwiseConvInt8: {
            auto& int8Params = param->symmetricQuan;
            for (int i = 0; i < int8Params->weight.size(); i++) {
                weightData.emplace_back(float(int8Params->weight[i]));
            }
            scales.resize(kernelNum, 1.0f);

            break;
        }
        default:
            break;
    }

    if (opType == MNN::OpType_ConvInt8 || opType == MNN::OpType_DepthwiseConvInt8) {
        param->quanParameter = IDSTEncoder::encode(weightData, scales, kernelSize, kernelNum, false, param->symmetricQuan->weight.data(), int(clampMin));
        param->symmetricQuan->weight.clear();
        param->quanParameter->alpha = {1.0f}; // fake scales
    } else {
        param->quanParameter = IDSTEncoder::encode(weightData, scales, kernelSize, kernelNum, asymmetricQuantFlag, quantWeights.data(), int(clampMin));
        param->weight.clear();
    }
};

void weightQuantAndCoding(std::unique_ptr<MNN::NetT>& netT, const modelConfig& config) {
    for (auto& op : netT->oplists) {
        WeightQuantAndCoding(op, config);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            WeightQuantAndCoding(op, config);
        }
    }
}
