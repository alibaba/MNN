//
//  WeightQuantAndCoding.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include "core/CommonCompute.hpp"
#include "core/IDSTEncoder.hpp"

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

    if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise &&
        opType != MNN::OpType_Deconvolution && opType != MNN::OpType_DeconvolutionDepthwise &&
        opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8) {
            return;
    }
    auto param           = op->main.AsConvolution2D();
    auto& common = param->common;
    if (param->quanParameter.get() != nullptr) {
        return;
    }

    if (config.weightQuantBits == 0) {
        if (opType == MNN::OpType_ConvInt8 || opType == MNN::OpType_DepthwiseConvInt8) {
            // Do nothing
        } else {
            CommonCompute::compressFloatWeightToSparse(op.get());
            return;
        }
    }
    int bits = 8;
    if ((config.weightQuantBits > 0) && (
        opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8)) {
        bits = config.weightQuantBits;
    }
    // Bits must from 2-8
    bits = std::max(bits, 2);
    bits = std::min(bits, 8);

    int weightSize = param->weight.size();
    // shared weights or sth else.
    if (weightSize == 0) {
        return;
    }
    if (opType == MNN::OpType_ConvInt8 || opType == MNN::OpType_DepthwiseConvInt8) {
        weightSize = param->symmetricQuan->weight.size();
    }
    int kernelNum = common->outputCount;
    int kernelSize = weightSize / kernelNum;
    int kxky = common->kernelX * common->kernelY;
    int icCount = kernelSize / kxky;

    bool asymmetricQuantFlag = config.weightQuantAsymmetric;

    float threshold = (float)(1 << (bits - 1)) - 1.0f;
    float clampMin = -threshold;
    if (asymmetricQuantFlag) {
        clampMin = -threshold - 1;
    }
    std::vector<float> weightData, scales;
    // block-wise quant
    int block_size = kernelSize, block_num = 1;
    if (config.weightQuantBlock > 0 && (kernelSize % config.weightQuantBlock == 0) && kxky == 1) {
        block_size = config.weightQuantBlock;
        block_num = kernelSize / block_size;
    }

    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_ConvolutionDepthwise:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_DeconvolutionDepthwise: {
            weightData = std::move(param->weight);
            if (asymmetricQuantFlag) {
                scales.resize(kernelNum * block_num * 2);
                for (int k = 0; k < kernelNum; k++) {
                    for (int b = 0; b < block_num; b++) {
                        int beginIndex = k * kernelSize + b * block_size;
                        auto minAndMax = findMinMax(weightData.data() + beginIndex, block_size);
                        float min = minAndMax[0];
                        float max = minAndMax[1];
                        float scale = (max - min) / (threshold - clampMin);

                        int scaleIndex = k * block_num + b;
                        scales[2 * scaleIndex] = min;
                        scales[2 * scaleIndex + 1] = scale;
                    }
                }
            } else {
                scales.resize(kernelNum * block_num);
                for (int k = 0; k < kernelNum; k++) {
                    for (int b = 0; b < block_num; b++) {
                        int beginIndex = k * kernelSize + b * block_size;
                        auto absMax = findAbsMax(weightData.data() + beginIndex, block_size);
                        int scaleIndex = k * block_num + b;
                        scales[scaleIndex] = absMax / threshold;
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

    kernelSize = block_size;
    kernelNum = kernelNum * block_num;
    if (opType == MNN::OpType_ConvInt8 || opType == MNN::OpType_DepthwiseConvInt8) {
        param->quanParameter = IDSTEncoder::encode(weightData.data(), scales, kernelSize, kernelNum, false, param->symmetricQuan->weight.data(), int(clampMin), bits);
        param->symmetricQuan->weight.clear();
        param->quanParameter->alpha = {1.0f}; // fake scales
    } else {
        param->quanParameter = IDSTEncoder::encode(weightData.data(), scales, kernelSize, kernelNum, asymmetricQuantFlag, nullptr, int(clampMin), bits, config.detectSparseSpeedUp);
        param->weight.clear();
        std::vector<float> empty;
        param->weight.swap(empty);
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
