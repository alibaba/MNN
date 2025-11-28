//
//  WeightQuantAndCoding.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include "HQQQuantizer.hpp"
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

static MNN::Quantization::HQQQuantizer::QuantizationResult _HQQQuant(const std::vector<float>& weights, int weightQuantBits, int weightQuantBlock, bool asymmetricQuantFlag) {
    MNN::Quantization::HQQQuantizer::QuantizationConfig hqqConfig;
    hqqConfig.bits = weightQuantBits;
    hqqConfig.group_size = weightQuantBlock;
    MNN::Quantization::HQQQuantizer hqq(hqqConfig);
    auto res = hqq.quantize(weights);
#if 0
    auto dequantized_weights = hqq.dequantize(res);
    // 计算量化误差
    float mse = 0.0f;
    float max_abs_error = 0.0f;

    for (size_t i = 0; i < weights.size(); ++i) {
        float error = weights[i] - dequantized_weights->readMap<float>()[i];
        float abs_error = std::abs(error);

        mse += error * error;
        max_abs_error = std::max(max_abs_error, abs_error);
    }

    mse /= weights.size();
    float rmse = std::sqrt(mse);

    std::cout << "量化误差分析:" << std::endl;
    std::cout << "  均方误差 (MSE): " << mse << std::endl;
    std::cout << "  均方根误差 (RMSE): " << rmse << std::endl;
    std::cout << "  最大绝对误差: " << max_abs_error << std::endl;
#endif
    return res;

}

void WeightQuantAndCoding(std::unique_ptr<MNN::OpT>& op, const modelConfig& config, const PostTreatContext* context) {
    const auto opType = op->type;
    // config.weightQuantBits only control weight quantization for float convolution
    // by default, do coding for convint8 and depthwiseconvint8, if there is any

    if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise &&
        opType != MNN::OpType_Deconvolution && opType != MNN::OpType_DeconvolutionDepthwise &&
        opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8) {
            return;
    }
    auto param = op->main.AsConvolution2D();
    auto& common = param->common;
    if (param->quanParameter.get() != nullptr) {
        return;
    }
    bool useHqq = config.useHQQ;
    auto weightQuantBits = config.weightQuantBits;
    bool asymmetricQuantFlag = config.weightQuantAsymmetric;
    auto weightQuantBlock = config.weightQuantBlock;
    // Read or write config in proto
    if (context->quantInfo.find(std::make_pair(context->subgraph, op->name)) != context->quantInfo.end()) {
        auto param = context->quantInfo.find(std::make_pair(context->subgraph, op->name))->second;
        if (param->weight_size() > 0) {
            auto weight = param->weight(0);
            if (weight.has_asymmetric()) {
                asymmetricQuantFlag = weight.asymmetric();
            }
            if (weight.has_bits()) {
                weightQuantBits = weight.bits();
            }
            if (weight.has_block_size()) {
                weightQuantBlock = weight.block_size();
            }
        }
    }
    if (useHqq) {
        // HQQ must use asym
        asymmetricQuantFlag = true;
    }
    if (nullptr != context->quantMutableInfo) {
        auto& proto = context->proto;
        auto layer = context->quantMutableInfo->add_layer();
        layer->set_op_name(op->name);
        if (!context->subgraph.empty()) {
            layer->set_subgraph_name(context->subgraph);
        }
        auto conv = layer->mutable_conv();
        conv->set_input_channel(common->inputCount);
        conv->set_output_channel(common->outputCount);
        conv->clear_kernel_size();
        conv->add_kernel_size(common->kernelX);
        conv->add_kernel_size(common->kernelY);
        auto weight = layer->add_weight();
        weight->set_bits(weightQuantBits);
        weight->set_asymmetric(asymmetricQuantFlag);
        weight->set_block_size(weightQuantBlock);
        weight->set_name(op->name);
    }

    if (weightQuantBits == 0) {
        if (opType == MNN::OpType_ConvInt8 || opType == MNN::OpType_DepthwiseConvInt8) {
            // Do nothing
        } else {
            CommonCompute::compressFloatWeightToSparse(op.get());
            return;
        }
    }
    int bits = 8;
    if ((weightQuantBits > 0) && (
        opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8)) {
        bits = weightQuantBits;
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
    int oc = common->outputCount;
    int kernelSize = weightSize / oc;
    int kxky = common->kernelX * common->kernelY;
    int icCount = kernelSize / kxky;

    float threshold = (float)(1 << (bits - 1)) - 1.0f;
    float clampMin = -threshold;
    if (asymmetricQuantFlag) {
        clampMin = -threshold - 1;
    }
    std::vector<float> weightData, scales;
    // block-wise quant
    int block_size = kernelSize, block_num = 1;
    if (weightQuantBlock > 0 && (icCount % weightQuantBlock == 0) && weightQuantBlock >= 16 && (weightQuantBlock % 16 == 0)) {
        block_num = common->inputCount / weightQuantBlock;
        block_size = weightQuantBlock * kxky;
    } else if (weightQuantBlock > 0 && (kernelSize % weightQuantBlock > 0)) {
        MNN_PRINT("weightQuantBlock=%d, inputChannel=%d: don't use block-quant for the layer: %s.\n", weightQuantBlock, icCount, op->name.c_str());
    } else if (weightQuantBlock > 0 && kxky > 1) {
        MNN_PRINT("The method of block quantization is not adopted to the layer: %s, because (kernel_x*kernel_y>1).\n", op->name.c_str());
    } else {
        // pass
    }
    MNN::Quantization::HQQQuantizer::QuantizationResult hqqRes;
    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_ConvolutionDepthwise:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_DeconvolutionDepthwise: {
            weightData = std::move(param->weight);
            if (useHqq) {
                hqqRes = _HQQQuant(weightData, bits, block_size, asymmetricQuantFlag);
                break;
            }
            if (asymmetricQuantFlag) {
                scales.resize(oc * block_num * 2);
                for (int k = 0; k < oc; k++) {
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
                scales.resize(oc * block_num);
                for (int k = 0; k < oc; k++) {
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
            scales.resize(oc, 1.0f);

            break;
        }
        default:
            break;
    }
    if (useHqq) {
        std::vector<float> mergeScale(hqqRes.SZ->getInfo()->size);
        ::memcpy(mergeScale.data(), hqqRes.SZ->readMap<float>(), mergeScale.size() * sizeof(float));
        param->quanParameter = IDSTEncoder::encode(nullptr, mergeScale, block_size, oc * block_num, true, hqqRes.QW->readMap<int8_t>(), int(clampMin), bits, false);
        param->weight.clear();
        std::vector<float> empty;
        param->weight.swap(empty);
    } else {
        if (opType == MNN::OpType_ConvInt8 || opType == MNN::OpType_DepthwiseConvInt8) {
            param->quanParameter = IDSTEncoder::encode(weightData.data(), scales, block_size, oc * block_num, false, param->symmetricQuan->weight.data(), int(clampMin), bits);
            param->symmetricQuan->weight.clear();
            param->quanParameter->alpha = {1.0f}; // fake scales
        } else {
            param->quanParameter = IDSTEncoder::encode(weightData.data(), scales, block_size, oc * block_num, asymmetricQuantFlag, nullptr, int(clampMin), bits, config.detectSparseSpeedUp);
            param->weight.clear();
            std::vector<float> empty;
            param->weight.swap(empty);
        }
    }

};
