//
//  LoRA.cpp
//  MNN
//
//  Created by MNN on 2024/03/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cstdlib>
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <chrono>

#include <string.h>
#include <stdlib.h>
#include <MNN/MNNDefine.h>
#include "LoRA.hpp"
#include "core/CommonCompute.hpp"
#include "core/MemoryFormater.h"
#include "core/IDSTDecoder.hpp"
#include "core/IDSTEncoder.hpp"
#include "core/ConvolutionCommon.hpp"

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>

int SymmetricQuantizeWeight(const float* weight, const int size, int8_t* quantizedWeight, float* scale,
                            const int channels, float weightClampValue) {
    const int channelStride     = size / channels;
    const int quantizedMaxValue = weightClampValue;

    for (int c = 0; c < channels; ++c) {
        const auto weightChannelStart    = weight + c * channelStride;
        auto quantizedWeightChannelStart = quantizedWeight + c * channelStride;
        auto minmaxValue                 = std::minmax_element(weightChannelStart, weightChannelStart + channelStride);
        const float dataAbsMax           = std::fmax(std::fabs(*minmaxValue.first), std::fabs(*minmaxValue.second));

        float scaleDataToInt8 = 1.0f;
        if (dataAbsMax == 0) {
            scale[c] = 0.0f;
        } else {
            scale[c]        = dataAbsMax / quantizedMaxValue;
            scaleDataToInt8 = quantizedMaxValue / dataAbsMax;
        }

        for (int i = 0; i < channelStride; ++i) {
            const int32_t quantizedInt8Value = static_cast<int32_t>(roundf(weightChannelStart[i] * scaleDataToInt8));
            quantizedWeightChannelStart[i] =
                std::min(quantizedMaxValue, std::max(-quantizedMaxValue, quantizedInt8Value));
        }
    }

    return 0;
}

std::unique_ptr<MNN::NetT> LoRA::load_model(const char* name) {
    std::ifstream inputFile(name, std::ios::binary);
    inputFile.seekg(0, std::ios::end);
    const auto size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    inputFile.read(buffer, size);
    inputFile.close();
    auto net = MNN::UnPackNet(buffer);
    delete[] buffer;
    MNN_ASSERT(net->oplists.size() > 0);
    return net;
}

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


LoRA::LoRA(const char* origin_model, const char* lora_model) {
    mMNNNet = std::move(load_model(origin_model));
    mLoRANet = std::move(load_model(lora_model));
    mExternalFile.reset(new std::fstream(std::string(origin_model) + ".weight", std::ios::in | std::ios::out | std::ios::binary));
    if (mExternalFile->bad()) {
        mExternalFile.reset(nullptr);
    }
}

LoRA::~LoRA() {
}

std::vector<std::string> split(const std::string& name, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(name);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

inline MNN::Express::VARP OpT2Const(MNN::OpT* op) {
    return MNN::Express::Variable::create(MNN::Express::Expr::create(op, {}, 1));
}

inline MNN::Express::VARP computeLoRA(MNN::OpT *lora_A, MNN::OpT *lora_B) {
    auto A = MNN::Express::_Cast(OpT2Const(lora_A), halide_type_of<float>());
    auto B = MNN::Express::_Cast(OpT2Const(lora_B), halide_type_of<float>());
    auto scale = MNN::Express::_Scalar<float>(4.0 * 5);
    auto lora = MNN::Express::_Multiply(MNN::Express::_MatMul(B, A), scale);
    // lora = MNN::Express::_Transpose(lora, {1, 0});
    return lora;
}

void LoRA::apply_external(MNN::OpT* op, MNN::OpT* lora_A, MNN::OpT* lora_B) {
    // lora origin weight
    auto result = std::make_shared<MNN::ConvolutionCommon::Int8Common>();
    auto param = op->main.AsConvolution2D();
    int ic = param->common->inputCount;
    int oc = param->common->outputCount;
    auto buffer_size = param->external[1];
    auto alpha_size = param->external[2];
    result->weight.reset(buffer_size);
    result->alpha.reset(alpha_size / sizeof(float));
    mExternalFile->seekg(param->external[0]);
    mExternalFile->read(reinterpret_cast<char*>(result->weight.get()), buffer_size);
    mExternalFile->read(reinterpret_cast<char*>(result->alpha.get()), alpha_size);
    auto& quan = param->quanParameter;
    size_t weightLength = 0;
    auto ptr = reinterpret_cast<unsigned char*>(result->weight.get());
    auto new_ptr = IDSTDecoder::ReadQuanData_c(ptr, &weightLength, result.get(), quan->shapeInt32);
    result->weight.set(new_ptr, weightLength);
    result->weightFloat.reset(weightLength);
    // dequant to float
    bool oldType4 = (quan->type == 4 && quan->aMin == 0 && std::abs(quan->quantScale) < 1e-6);
    if (quan->readType != 0 || oldType4) {
        result->asymmetric = true;
        float clampMin = quan->aMin == 0 ? -128 : quan->aMin;
        for (int o = 0; o < oc; ++o) {
            float min = result->alpha.get()[2 * o];
            float alpha = result->alpha.get()[2 * o + 1];
            min = min - clampMin * alpha;
            auto dstW   = result->weightFloat.get() + o * ic;
            auto srcW   = result->weight.get() + o * ic;
            for (int v=0; v < ic; ++v) {
                dstW[v] = (float)srcW[v] * alpha + min;
            }
        }
    } else {
        result->asymmetric = false;
        for (int o = 0; o < oc; ++o) {
            float alpha = result->alpha.get()[o];
            auto dstW   = result->weightFloat.get() + o * ic;
            auto srcW   = result->weight.get() + o * ic;
            for (int v=0; v < ic; ++v) {
                dstW[v] = (float)srcW[v] * alpha;
            }
        }
    }
    result->weight.release();
    result->alpha.release();
    auto weight = Express::_Const(result->weightFloat.get(), {oc, ic});
    auto lora = computeLoRA(lora_A, lora_B);
    result->weightFloat.release();
    weight = Express::_Add(weight, lora);
    // weight = Express::_Subtract(weight, lora);
    // quant
    int bits = 4;
    float threshold = (float)(1 << (bits - 1)) - 1.0f;
    auto clampMin = quan->aMin;
    std::vector<float> scales;
    std::vector<int8_t> quantWeights;
    if (result->asymmetric) {
        scales.resize(oc*2);
        for (int o = 0; o < oc; ++o) {
            const float* ptr = weight->readMap<float>() + o * ic;
            auto minAndMax = findMinMax(ptr, ic);
            float min = minAndMax[0];
            float max = minAndMax[1];
            float scale = (max - min) / (threshold - clampMin);

            scales[2*o] = min;
            scales[2*o+1] = scale;
            /*
            for (int ii = 0; ii < partWeightSize; ii++) {
                int8_t quantValue = int8_t(std::round((ptr[ii] - min) / scale + clampMin));
                quantWeights.emplace_back(quantValue);
            }
            */
        }
    }
    auto res = IDSTEncoder::encode(weight->readMap<float>(), scales, ic, oc, result->asymmetric, /*quantWeights.data()*/nullptr, int(clampMin), bits, false);
    mExternalFile->seekp(param->external[0]);
    mExternalFile->write(reinterpret_cast<char*>(res->buffer.data()), buffer_size);
    mExternalFile->write(reinterpret_cast<char*>(res->alpha.data()), alpha_size);
}

void LoRA::apply_lora() {
    std::set<std::string> lora_keys;
    std::map<std::string, std::pair<OpT*, OpT*>> loras;
    for (int i = 0; i < mLoRANet->oplists.size(); i+= 2) {
        auto& op_A = mLoRANet->oplists[i];
        auto& op_B = mLoRANet->oplists[i + 1];
        auto tokens = split(op_A->name, '/');
        auto layer = tokens[4];
        auto key = tokens[6];
        lora_keys.insert(key);
        loras[layer + key] = std::make_pair(op_A.get(), op_B.get());
    }
    for (auto& op : mMNNNet->oplists) {
        if (op->type == MNN::OpType_Convolution) {
            bool has_lora = false;
            for (auto key : lora_keys) {
                if (op->name.find(key) != std::string::npos) {
                    has_lora = true;
                    break;
                }
            }
            if (!has_lora) continue;
            auto tokens = split(op->name, '/');
            auto layer = split(tokens[1], '.')[1];
            auto key = tokens[3];
            auto lora = loras[layer + key];
            apply_external(op.get(), lora.first, lora.second);
        }
    }
    mExternalFile->flush();
    mExternalFile->close();
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./LoRA ${origin.mnn} ${lora.mnn}\n");
        return 0;
    }
    const char* origin_model = argv[1];
    const char* lora_model = argv[2];
    auto lora = std::unique_ptr<LoRA>(new LoRA(origin_model, lora_model));
    auto st = std::chrono::system_clock::now();
    lora->apply_lora();
    auto et = std::chrono::system_clock::now();
    auto lora_during = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count() * 1e-6;
    printf("### total time = %.2f s\n", lora_during);
    return 0;
}
