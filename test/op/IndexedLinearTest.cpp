//
//  IndexedLinearTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <MNN/MNNDefine.h>
#include "core/Backend.hpp"
#include "core/IDSTEncoder.hpp"
#include "core/IndexedLinearCommon.hpp"
#include "module/StaticModule.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <limits>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

class IndexedLinearTest : public MNNTestCase {
    const IndexedLinearParam* indexedParameter(const Module* module) const {
        if (module->type() == "StaticModule") {
            auto session = static_cast<const StaticModule*>(module)->getSession();
            for (const auto& cache : session->getPipelineInfo(0).second) {
                for (const auto& execution : cache.executionCache) {
                    if (execution.first->type() == OpType_IndexedLinear) {
                        return execution.first->main_as_IndexedLinearParam();
                    }
                }
            }
        }
        for (const auto& child : module->getChildren()) {
            auto param = indexedParameter(child.get());
            if (param) {
                return param;
            }
        }
        return nullptr;
    }

    bool validWeightLengths() const {
        for (int elements : {1, 8, 9}) {
            OpT op;
            op.type = OpType_IndexedLinear;
            op.main.type = OpParameter_IndexedLinearParam;
            auto param = new IndexedLinearParamT;
            op.main.value = param;
            std::unique_ptr<Convolution2DT> expert(new Convolution2DT);
            expert->common.reset(new Convolution2DCommonT);
            expert->common->inputCount = 4;
            expert->common->outputCount = 2;
            expert->quanParameter.reset(new IDSTQuanT);
            expert->quanParameter->type = 3;
            expert->quanParameter->buffer.resize(elements * sizeof(uint16_t), 0);
            param->linears.emplace_back(std::move(expert));
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(Op::Pack(builder, &op));
            auto weight =
                IndexedLinearCommon::loadLinear(flatbuffers::GetRoot<Op>(builder.GetBufferPointer()), 0, nullptr);
            if (static_cast<bool>(weight) != (elements == 8)) {
                MNN_ERROR("IndexedLinear accepted invalid FP16 bank length: %d\n", elements);
                return false;
            }
            if (elements == 8) {
                if (param->fusedActivation != -1) {
                    return false;
                }
                for (int activation : {-1, static_cast<int>(UnaryOpOperation_SILU), 0, 1,
                                       static_cast<int>(UnaryOpOperation_GELU), -2, 999}) {
                    param->fusedActivation = activation;
                    builder.Clear();
                    builder.Finish(Op::Pack(builder, &op));
                    int count, inputs, outputs;
                    bool valid = IndexedLinearCommon::dimensions(flatbuffers::GetRoot<Op>(builder.GetBufferPointer()),
                                                                 count, inputs, outputs);
                    if (valid != (activation == -1 || activation == UnaryOpOperation_SILU)) {
                        MNN_ERROR("IndexedLinear invalid fused UnaryOpOperation: %d\n", activation);
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool fusedSiluRounding() const {
        const int channels = 8;
        OpT op;
        op.type = OpType_IndexedLinear;
        op.main.type = OpParameter_IndexedLinearParam;
        auto param = new IndexedLinearParamT;
        op.main.value = param;
        std::unique_ptr<Convolution2DT> expert(new Convolution2DT);
        expert->common.reset(new Convolution2DCommonT);
        expert->common->inputCount = 64;
        expert->common->outputCount = 2 * channels;
        expert->weight.resize(64 * 2 * channels, 0.0f);
        const float gates[] = {-12.0f, -11.0859375f, -8.953125f, -3.375f, -0.5f, 0.25f, 2.75f, 8.0f};
        for (int i = 0; i < channels; ++i) {
            expert->weight[i * 64] = gates[i];
            expert->weight[(channels + i) * 64] = 1024.0f;
        }
        param->linears.emplace_back(std::move(expert));
        auto x = _Input({1, 64}, NCHW);
        std::fill(x->writeMap<float>(), x->writeMap<float>() + 64, 1.0f);
        auto ids = _Input({1, 1}, NCHW, halide_type_of<int32_t>());
        ids->writeMap<int32_t>()[0] = 0;
        auto raw = Variable::create(Expr::create(&op, {x, ids}));
        auto parts = _Split(raw, {channels, channels}, 2);
        auto reference = _Multiply(_Silu(parts[0]), parts[1]);
        param->fusedActivation = UnaryOpOperation_SILU;
        auto fused = Variable::create(Expr::create(&op, {x, ids}));
        auto expected = reference->readMap<float>();
        auto actual = fused->readMap<float>();
        if (!expected || !actual)
            return false;
        for (int i = 0; i < channels; ++i) {
            if (expected[i] != actual[i]) {
                MNN_ERROR("IndexedLinear fused SiLU changed rounding at %d: %f != %f\n", i, actual[i], expected[i]);
                return false;
            }
        }
        return true;
    }

    bool fusedFloatingBanks() const {
        for (int layout : {1, 2, 3}) {
            OpT op;
            op.type = OpType_IndexedLinear;
            op.main.type = OpParameter_IndexedLinearParam;
            auto param = new IndexedLinearParamT;
            op.main.value = param;
            const int channels = 3;
            std::unique_ptr<Convolution2DT> conv(new Convolution2DT);
            conv->common.reset(new Convolution2DCommonT);
            conv->common->inputCount = 64;
            conv->common->outputCount = 2 * channels;
            conv->weight.resize(64 * 2 * channels, 0.0f);
            for (int projection = 0; projection < 2; ++projection) {
                for (int channel = 0; channel < channels; ++channel) {
                    float value = 1.00024f + channel * 0.25f;
                    if (layout & (1 << projection)) {
                        value = static_cast<float>(half_float::half(value));
                    }
                    conv->weight[(projection * channels + channel) * 64] = value;
                }
            }
            // A merged FP32 bank preserves both original projection precisions.
            // When both projections are FP16, the merged bank remains FP16.
            if (layout == 3) {
                conv->quanParameter.reset(new IDSTQuanT);
                conv->quanParameter->type = 3;
                conv->quanParameter->buffer.resize(conv->weight.size() * sizeof(uint16_t));
                for (int index = 0; index < conv->weight.size(); ++index) {
                    half_float::half value(conv->weight[index]);
                    ::memcpy(conv->quanParameter->buffer.data() + index * sizeof(uint16_t), &value, sizeof(uint16_t));
                }
                conv->weight.clear();
            }
            param->linears.emplace_back(std::move(conv));
            for (int tokens : {1, 7}) {
                auto x = _Input({tokens, 64}, NCHW);
                std::fill(x->writeMap<float>(), x->writeMap<float>() + tokens * 64, 1.0f);
                auto ids = _Input({tokens, 1}, NCHW, halide_type_of<int32_t>());
                std::fill(ids->writeMap<int32_t>(), ids->writeMap<int32_t>() + tokens, 0);
                param->fusedActivation = -1;
                auto raw = Variable::create(Expr::create(&op, {x, ids}));
                auto parts = _Split(raw, {channels, channels}, 2);
                auto reference = _Multiply(_Silu(parts[0]), parts[1]);
                param->fusedActivation = UnaryOpOperation_SILU;
                auto fused = Variable::create(Expr::create(&op, {x, ids}));
                auto expected = reference->readMap<float>();
                auto actual = fused->readMap<float>();
                if (!expected || !actual)
                    return false;
                for (int index = 0; index < tokens * channels; ++index) {
                    const float tolerance =
                        4.0f * std::numeric_limits<float>::epsilon() * (1.0f + std::abs(expected[index]));
                    if (!std::isfinite(actual[index]) || std::abs(expected[index] - actual[index]) > tolerance) {
                        MNN_ERROR("IndexedLinear floating banks layout=%d T=%d index=%d: %f != %f\n", layout, tokens,
                                  index, actual[index], expected[index]);
                        return false;
                    }
                }
            }
        }
        return true;
    }

    const Execution* indexedExecution(const Module* module) const {
        if (module->type() == "StaticModule") {
            auto session = static_cast<const StaticModule*>(module)->getSession();
            for (const auto& cache : session->getPipelineInfo(0).second) {
                for (const auto& execution : cache.executionCache) {
                    if (execution.first->type() == OpType_IndexedLinear) {
                        return execution.second.get();
                    }
                }
            }
        }
        for (const auto& child : module->getChildren()) {
            auto execution = indexedExecution(child.get());
            if (execution) {
                return execution;
            }
        }
        return nullptr;
    }

    bool runCase(int tokens, int topK, int inputChannels, int outputChannels, int inputMode, int precision,
                 int weightBits = 0, int blockChannels = 0, bool moduleTest = false, bool external = false,
                 int dynamicQuantOption = 0, int scaleBits = 32, int experts = 5,
                 const std::vector<int>& tokenSequence = {}, bool fuseSwiGLU = false) {
        const bool resizeTest = !tokenSequence.empty();
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_IndexedLinear;
        op->main.type = OpParameter_IndexedLinearParam;
        auto parameter = new IndexedLinearParamT;
        op->main.value = parameter;
        parameter->inputMode = inputMode;
        parameter->fusedActivation = fuseSwiGLU ? UnaryOpOperation_SILU : -1;
        const int resultChannels = fuseSwiGLU ? outputChannels / 2 : outputChannels;
        std::vector<std::vector<float>> weights(experts), biases(experts);
        const std::string externalName =
            "indexed_linear_test_" +
            std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + ".weight";
        const char* externalPath = externalName.c_str();
        std::ofstream externalFile;
        struct RemoveExternal {
            const char* path;
            ~RemoveExternal() {
                if (path) {
                    std::remove(path);
                }
            }
        } removeExternal{external ? externalPath : nullptr};
        if (external) {
            externalFile.open(externalPath, std::ios::binary | std::ios::trunc);
            if (!externalFile) {
                return false;
            }
            op->externalPath = externalPath;
        }
        for (int expert = 0; expert < experts; ++expert) {
            std::unique_ptr<Convolution2DT> conv(new Convolution2DT);
            conv->common.reset(new Convolution2DCommonT);
            conv->common->inputCount = inputChannels;
            conv->common->outputCount = outputChannels;
            conv->common->kernelX = conv->common->kernelY = 1;
            conv->common->strideX = conv->common->strideY = 1;
            conv->common->dilateX = conv->common->dilateY = 1;
            conv->common->group = 1;
            weights[expert].resize(inputChannels * outputChannels);
            biases[expert].resize(outputChannels);
            for (int index = 0; index < weights[expert].size(); ++index) {
                weights[expert][index] = std::sin(float(index * 3 + expert * 7)) * 0.08f;
            }
            for (int channel = 0; channel < outputChannels; ++channel) {
                biases[expert][channel] = float((channel + expert) % 9 - 4) * 0.025f;
            }
            conv->weight = weights[expert];
            conv->bias = biases[expert];
            if (weightBits) {
                const int blocks = inputChannels / blockChannels;
                const int codeRange = 1 << weightBits;
                const int minimum = -codeRange / 2;
                std::vector<int8_t> codes(inputChannels * outputChannels);
                std::vector<float> alpha(2 * outputChannels * blocks);
                for (int channel = 0; channel < outputChannels; ++channel) {
                    for (int block = 0; block < blocks; ++block) {
                        const int index = channel * blocks + block;
                        const float scale = (0.075f + float((channel + block + expert) % 3) * 0.0075f) / codeRange;
                        const float offset = -0.043f + float((channel + 3 * block + expert) % 5) * 0.004f;
                        alpha[2 * index] = offset;
                        alpha[2 * index + 1] = scale;
                        for (int inner = 0; inner < blockChannels; ++inner) {
                            const int inputIndex = channel * inputChannels + block * blockChannels + inner;
                            const int code = (inputIndex * 11 + channel * 7 + expert * 3) % codeRange;
                            codes[inputIndex] = minimum + code;
                            weights[expert][inputIndex] = offset + code * scale;
                        }
                    }
                }
                // Mix current signed-code metadata with legacy minimum-code metadata in one bank.
                // FP16 packing must preserve the latter's FP32 correction instead of re-rounding it.
                const bool signedAlpha = scaleBits == 16 && expert % 2 == 0;
                if (signedAlpha) {
                    for (int index = 0; index < alpha.size() / 2; ++index) {
                        alpha[2 * index] -= minimum * alpha[2 * index + 1];
                    }
                }
                conv->quanParameter = IDSTEncoder::encode(nullptr, alpha, blockChannels, outputChannels * blocks, true,
                                                          codes.data(), minimum, {weightBits, false, scaleBits});
                if (signedAlpha) {
                    conv->quanParameter->aMin = 1; // Current converter format: alpha already addresses signed codes.
                }
                if (scaleBits == 16) {
                    for (int index = 0; index < alpha.size(); ++index) {
                        half_float::half value;
                        ::memcpy(&value, &conv->quanParameter->alphaFp16[index], sizeof(uint16_t));
                        alpha[index] = static_cast<float>(value);
                    }
                    for (int index = 0; index < weights[expert].size(); ++index) {
                        const int block = index / blockChannels;
                        weights[expert][index] =
                            alpha[2 * block] + (int(codes[index]) - (signedAlpha ? 0 : minimum)) * alpha[2 * block + 1];
                    }
                }
                conv->weight.clear();
            }
            if (external) {
                const auto offset = static_cast<int64_t>(externalFile.tellp());
                if (weightBits) {
                    auto& quant = *conv->quanParameter;
                    const int64_t weightBytes = quant.buffer.size();
                    const int64_t scaleBytes = scaleBits == 16 ? quant.alphaFp16.size() * sizeof(uint16_t)
                                                               : quant.alpha.size() * sizeof(float);
                    externalFile.write(reinterpret_cast<const char*>(quant.buffer.data()), weightBytes);
                    externalFile.write(scaleBits == 16 ? reinterpret_cast<const char*>(quant.alphaFp16.data())
                                                       : reinterpret_cast<const char*>(quant.alpha.data()),
                                       scaleBytes);
                    conv->external = {offset, weightBytes, scaleBytes,
                                      static_cast<int64_t>(conv->bias.size() * sizeof(float))};
                    quant.buffer.clear();
                    quant.alpha.clear();
                    quant.alphaFp16.clear();
                } else {
                    externalFile.write(reinterpret_cast<const char*>(conv->weight.data()),
                                       conv->weight.size() * sizeof(float));
                    conv->external = {offset, static_cast<int64_t>(conv->weight.size() * sizeof(float)),
                                      static_cast<int64_t>(conv->bias.size() * sizeof(float))};
                    conv->weight.clear();
                }
                externalFile.write(reinterpret_cast<const char*>(conv->bias.data()), conv->bias.size() * sizeof(float));
                conv->bias.clear();
            }
            parameter->linears.emplace_back(std::move(conv));
        }
        if (external) {
            externalFile.close();
        }
        auto input = _Input(inputMode == 0 ? std::vector<int>{tokens, inputChannels}
                                           : std::vector<int>{tokens, topK, inputChannels},
                            NCHW);
        auto ids = _Input({tokens, topK}, NCHW, halide_type_of<int32_t>());
        input->setName("x");
        ids->setName("ids");
        std::vector<float> values;
        auto result = Variable::create(Expr::create(op.get(), {input, ids}));
        result->setName("y");
        std::shared_ptr<Module> module, cloned;
        if (moduleTest) {
            const auto status = MNNTestSuite::get()->pStaus;
            BackendConfig backendConfig;
            backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
            backendConfig.memory = BackendConfig::Memory_Low;
            ScheduleConfig schedule;
            schedule.type = static_cast<MNNForwardType>(status.forwardType);
            schedule.backendConfig = &backendConfig;
            schedule.numThread = status.thread > 0 ? status.thread : 1;
            std::shared_ptr<Executor::RuntimeManager> runtime(Executor::RuntimeManager::createRuntimeManager(schedule),
                                                              Executor::RuntimeManager::destroy);
            runtime->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, dynamicQuantOption);
            Module::Config config;
            config.rearrange = true;
            config.shapeMutable = true;
            auto buffer = Variable::save({result});
            module.reset(Module::load({"x", "ids"}, {"y"}, reinterpret_cast<const uint8_t*>(buffer.data()),
                                      buffer.size(), runtime, &config));
            if (!module) {
                return false;
            }
            if (!external) {
                auto bank = indexedParameter(module.get());
                if (!bank) {
                    return false;
                }
                for (auto experts : {bank->linears()}) {
                    if (!experts)
                        continue;
                    for (auto expert : *experts) {
                        auto quant = expert->quanParameter();
                        if ((expert->weight() && expert->weight()->size()) ||
                            (expert->bias() && expert->bias()->size()) ||
                            (quant && ((quant->buffer() && quant->buffer()->size()) ||
                                       (quant->alpha() && quant->alpha()->size()) ||
                                       (quant->alphaFp16() && quant->alphaFp16()->size()) ||
                                       (quant->index() && quant->index()->size())))) {
                            MNN_ERROR("IndexedLinear retained inline weights after rearrange\n");
                            return false;
                        }
                    }
                }
            }
            if (!resizeTest) {
                cloned.reset(Module::clone(module.get(), true));
                if (!cloned) {
                    return false;
                }
            }
            // Neither the caller's serialized buffer nor the original module is
            // needed by the cloned execution's packed immutable weight resources.
            buffer.clear();
        }
        // Ordinary cases also check clone lifetime. Shape cases keep the same
        // Module and assert that resize reuses the cached backend execution.
        const Execution* previousExecution = nullptr;
        const int passes = resizeTest ? static_cast<int>(tokenSequence.size()) : 2;
        for (int pass = 0; pass < passes; ++pass) {
            if (resizeTest) {
                tokens = tokenSequence[pass];
                if (!input->resize(inputMode == 0 ? std::vector<int>{tokens, inputChannels}
                                                  : std::vector<int>{tokens, topK, inputChannels}) ||
                    !ids->resize({tokens, topK})) {
                    return false;
                }
            }
            const int rows = inputMode == 0 ? tokens : tokens * topK;
            values.resize(rows * inputChannels);
            for (int index = 0; index < values.size(); ++index) {
                values[index] = std::cos(float(index * 7 + 2 + (resizeTest ? pass : 0))) * 0.25f;
            }
            ::memcpy(input->writeMap<float>(), values.data(), values.size() * sizeof(float));
            std::vector<int32_t> routes(tokens * topK);
            for (int route = 0; route < routes.size(); ++route) {
                routes[route] = resizeTest ? (route * 7 + route / topK + pass) % experts
                                           : (pass == 0 ? (route * 3 + route / topK) % (experts - 1) : experts - 1);
            }
            ::memcpy(ids->writeMap<int32_t>(), routes.data(), routes.size() * sizeof(int32_t));
            auto evaluated = result;
            if (moduleTest) {
                if (!resizeTest && pass == 1) {
                    module.reset();
                    if (external) {
                        std::remove(externalPath);
                    }
                }
                auto outputs = (resizeTest || pass == 0 ? module : cloned)->onForward({input, ids});
                if (outputs.size() != 1) {
                    return false;
                }
                evaluated = outputs[0];
                if (resizeTest) {
                    // The cache holds the compute execution. Pipeline inserts
                    // cross-backend copies as separate commands afterward.
                    auto execution = indexedExecution(module.get());
                    if (!execution || (previousExecution && execution != previousExecution)) {
                        MNN_ERROR("IndexedLinear dynamic-shape test did not reuse its execution\n");
                        return false;
                    }
                    const int requestedBackend = MNNTestSuite::get()->pStaus.forwardType;
                    const bool requireGPU =
                        requestedBackend == MNN_FORWARD_METAL || requestedBackend == MNN_FORWARD_CUDA ||
                        requestedBackend == MNN_FORWARD_OPENCL || requestedBackend == MNN_FORWARD_VULKAN;
                    const int actualBackend = execution->backend() ? execution->backend()->type() : -1;
                    if (requireGPU && actualBackend != requestedBackend) {
                        MNN_ERROR("IndexedLinear resize T=%d requested backend=%d, actual backend=%d\n", tokens,
                                  requestedBackend, actualBackend);
                        return false;
                    }
                    MNN_PRINT("IndexedLinear resize T=%d backend=%d sameExecution=1\n", tokens, actualBackend);
                    previousExecution = execution;
                }
            }
            const auto actual = evaluated->readMap<float>();
            if (!actual) {
                MNN_ERROR("IndexedLinear produced no output\n");
                return false;
            }
            if (evaluated->getInfo()->dim != std::vector<int>({tokens, topK, resultChannels})) {
                return false;
            }
            for (int route = 0; route < routes.size(); ++route) {
                const int expert = routes[route];
                const int sourceRow = inputMode == 0 ? route / topK : route;
                for (int channel = 0; channel < resultChannels; ++channel) {
                    double expected = biases[expert][channel];
                    for (int inner = 0; inner < inputChannels; ++inner) {
                        expected += values[sourceRow * inputChannels + inner] *
                                    weights[expert][channel * inputChannels + inner];
                    }
                    if (fuseSwiGLU) {
                        double up = biases[expert][channel + resultChannels];
                        for (int inner = 0; inner < inputChannels; ++inner) {
                            up += values[sourceRow * inputChannels + inner] *
                                  weights[expert][(channel + resultChannels) * inputChannels + inner];
                        }
                        expected = expected / (1.0 + std::exp(-expected)) * up;
                    }
                    const float tolerance = weightBits ? 0.006f : (precision == 2 ? 0.003f : 0.0001f);
                    if (!std::isfinite(actual[route * resultChannels + channel]) ||
                        std::abs(actual[route * resultChannels + channel] - expected) >
                            tolerance * (1.0 + std::abs(expected))) {
                        MNN_ERROR("IndexedLinear mode=%d T=%d K=%d I=%d O=%d pass=%d route=%d channel=%d: %f != %f\n",
                                  inputMode, tokens, topK, inputChannels, outputChannels, pass, route, channel,
                                  actual[route * resultChannels + channel], static_cast<float>(expected));
                        return false;
                    }
                }
            }
        }
        return true;
    }

public:
    bool run(int precision) override {
        if (!validWeightLengths()) {
            return false;
        }
        if (!fusedSiluRounding() || !fusedFloatingBanks())
            return false;
        // Sparse routing: most experts are empty, at most T*K input tiles exist.
        if (!runCase(2, 2, 31, 65, 0, precision, 0, 0, true, false, 0, 32, 257))
            return false;
        // E=60/K=4 crosses the GPU M32/M8 threshold in one execution. Although
        // routes shrink 516 -> 476, tile capacity grows 77 -> 120 and Vulkan's
        // combined workspace grows 791 -> 837 ints. Exercise the return to M32.
        if (!runCase(129, 4, 31, 65, 0, precision, 0, 0, true, false, 0, 32, 60, {129, 119, 129})) {
            return false;
        }
        if (!runCase(129, 4, 31, 130, 0, precision, 0, 0, true, false, 0, 32, 60, {129, 119, 1, 129}, true))
            return false;
        for (int tokens : {1, 7, 37, 577}) {
            if (!runCase(tokens, 4, 31, 134, 0, precision, 0, 0, true, true, 0, 32, 5, {}, true))
                return false;
            for (int bits : {4, 8}) {
                if (!runCase(tokens, 4, 128, 262, 0, precision, bits, 32, true, true, 0, 16, 5, {}, true))
                    return false;
            }
        }
        for (int mode : {0, 1}) {
            if (!runCase(1, 1, 3, 5, mode, precision) || !runCase(1, 4, 64, 135, mode, precision) ||
                !runCase(7, 2, 31, 67, mode, precision) || !runCase(37, 3, 64, 128, mode, precision)) {
                return false;
            }
            if (!runCase(7, 2, 32, 65, mode, precision, 0, 0, true, true)) {
                return false;
            }
            for (int bits : {4, 8}) {
                for (int block : {32, 64, 128}) {
                    for (int scaleBits : {16, 32}) {
                        if (!runCase(1, 4, 128, 131, mode, precision, bits, block, false, false, 0, scaleBits) ||
                            !runCase(37, 3, 128, 131, mode, precision, bits, block, false, false, 0, scaleBits)) {
                            return false;
                        }
                    }
                }
                if (!runCase(7, 2, 64, 65, mode, precision, bits, 32, true, true)) {
                    return false;
                }
                if (!runCase(7, 2, 64, 65, mode, precision, bits, 32, true, true, 0, 16)) {
                    return false;
                }
                for (int option : {1, 2}) {
                    if (!runCase(1, 4, 64, 65, mode, precision, bits, 32, true, false, option) ||
                        !runCase(13, 3, 64, 65, mode, precision, bits, 32, true, false, option)) {
                        return false;
                    }
                    if (!runCase(37, 3, 128, 131, mode, precision, bits, 32, true, true, option, 16, 5,
                                 {37, 1, 13, 1, 37})) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(IndexedLinearTest, "op/IndexedLinear");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
