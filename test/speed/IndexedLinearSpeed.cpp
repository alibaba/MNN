//
//  IndexedLinearSpeed.cpp
//  MNNTests
//
//  Created by MNN on 2026/10/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/IDSTEncoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

namespace {

struct IndexedShape {
    const char* tag;
    int tokens;
    int topK;
    int inputChannels;
    int outputChannels;
    int experts;
    int inputMode;
    int bits;
    int blockChannels;
    bool fuseSwiGLU;
    int routeWeight;
};

int indexedBits() {
    static const int value = [] {
        const char* env = getenv("MNN_IDX_BITS");
        if (nullptr != env) {
            const int bits = atoi(env);
            if (bits == 4 || bits == 8 || bits == 16) {
                return bits == 16 ? 0 : bits;
            }
        }
        return 4;
    }();
    return value;
}

int indexedTokens() {
    static const int value = [] {
        const char* env = getenv("MNN_IDX_TOKENS");
        if (nullptr != env) {
            const int tokens = atoi(env);
            if (tokens > 0) {
                return tokens;
            }
        }
        return 512;
    }();
    return value;
}

std::shared_ptr<Executor::RuntimeManager> indexedRuntime(int thread = 1) {
    auto status = MNNTestSuite::get()->pStaus;
    ScheduleConfig schedule;
    schedule.type = static_cast<MNNForwardType>(status.forwardType);
    schedule.numThread = thread > 0 ? thread : 1;
    BackendConfig config;
    config.precision = static_cast<BackendConfig::PrecisionMode>(status.precision);
    config.memory = BackendConfig::Memory_Low;
    schedule.backendConfig = &config;
    return std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(schedule));
}

VARP indexedFloatInput(const std::vector<int>& dims, bool deviceResident) {
    auto var = _Input(dims, NCHW);
    auto data = var->writeMap<float>();
    int size = 1;
    for (int dim : dims) {
        size *= dim;
    }
    for (int i = 0; i < size; ++i) {
        data[i] = std::sin(float(i * 17 + 3)) * 0.13f;
    }
    var->unMap();
    return deviceResident ? var * _Scalar<float>(1.0f) : var;
}

VARP indexedIds(const IndexedShape& shape) {
    auto ids = _Input({shape.tokens, shape.topK}, NCHW, halide_type_of<int32_t>());
    auto data = ids->writeMap<int>();
    for (int route = 0; route < shape.tokens * shape.topK; ++route) {
        data[route] = int((uint32_t)(route * 7919 + shape.topK * 13) % shape.experts);
    }
    ids->unMap();
    return ids;
}

std::unique_ptr<Convolution2DT> indexedExpert(int expert, int inputChannels, int outputChannels, int bits,
                                              int blockChannels) {
    std::unique_ptr<Convolution2DT> conv(new Convolution2DT);
    conv->common.reset(new Convolution2DCommonT);
    conv->common->inputCount = inputChannels;
    conv->common->outputCount = outputChannels;
    conv->common->kernelX = conv->common->kernelY = 1;
    conv->common->strideX = conv->common->strideY = 1;
    conv->common->dilateX = conv->common->dilateY = 1;
    conv->common->group = 1;

    std::vector<float> weight(inputChannels * outputChannels);
    std::vector<float> bias(outputChannels);
    for (int index = 0; index < static_cast<int>(weight.size()); ++index) {
        weight[index] = std::sin(float(index * 3 + expert * 7)) * 0.06f;
    }
    for (int channel = 0; channel < outputChannels; ++channel) {
        bias[channel] = float((channel + expert) % 9 - 4) * 0.02f;
    }
    if (bits == 0) {
        conv->weight = std::move(weight);
        conv->bias = std::move(bias);
        return conv;
    }

    const int blocks = inputChannels / blockChannels;
    const int codeRange = 1 << bits;
    const int minimum = -codeRange / 2;
    std::vector<int8_t> codes(weight.size());
    std::vector<float> alpha(2 * outputChannels * blocks);
    for (int channel = 0; channel < outputChannels; ++channel) {
        for (int block = 0; block < blocks; ++block) {
            const int group = channel * blocks + block;
            const float scale = (0.055f + float((channel + block + expert) % 3) * 0.006f) / codeRange;
            alpha[2 * group] = 0.0f;
            alpha[2 * group + 1] = scale;
            for (int inner = 0; inner < blockChannels; ++inner) {
                const int inputIndex = channel * inputChannels + block * blockChannels + inner;
                const int code = (inputIndex * 11 + channel * 7 + expert * 3) % codeRange;
                codes[inputIndex] = static_cast<int8_t>(minimum + code);
                weight[inputIndex] = code * scale;
            }
        }
    }
    conv->quanParameter = IDSTEncoder::encode(nullptr, alpha, blockChannels, outputChannels * blocks, true,
                                              codes.data(), minimum, {bits, false, 32});
    conv->bias = std::move(bias);
    return conv;
}

std::shared_ptr<Module> indexedModule(const IndexedShape& shape, VARP* input, VARP* ids, VARP* weights) {
    auto op = std::unique_ptr<OpT>(new OpT);
    op->type = OpType_IndexedLinear;
    op->main.type = OpParameter_IndexedLinearParam;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    auto parameter = new IndexedLinearParamT;
    op->main.value = parameter;
    parameter->inputMode = shape.inputMode;
    parameter->fusedActivation = shape.fuseSwiGLU ? UnaryOpOperation_SILU : -1;
    parameter->routeWeight = shape.routeWeight;
    for (int expert = 0; expert < shape.experts; ++expert) {
        parameter->linears.emplace_back(indexedExpert(expert, shape.inputChannels, shape.outputChannels, shape.bits,
                                                      shape.bits ? shape.blockChannels : 1));
    }

    const auto inputDims = shape.inputMode == 0 ? std::vector<int>{shape.tokens, shape.inputChannels}
                                                : std::vector<int>{shape.tokens, shape.topK, shape.inputChannels};
    *input = indexedFloatInput(inputDims, true);
    *ids = indexedIds(shape);
    (*input)->setName("x");
    (*ids)->setName("ids");
    std::vector<VARP> inputs = {*input, *ids};
    if (shape.routeWeight) {
        const auto wDims = (shape.routeWeight == 2 || shape.routeWeight == 3)
                               ? std::vector<int>{shape.tokens, shape.experts}
                               : std::vector<int>{shape.tokens, shape.topK};
        auto w = _Input(wDims, NCHW);
        auto wd = w->writeMap<float>();
        const int wSize = wDims[0] * wDims[1];
        for (int i = 0; i < wSize; ++i) {
            wd[i] = shape.routeWeight == 2 ? 0.1f + 0.05f * std::abs(std::sin(float(i * 17 + 3)))
                                           : std::sin(float(i * 17 + 3)) * 0.13f;
        }
        w->unMap();
        *weights = w * _Scalar<float>(1.0f);
        (*weights)->setName("w");
        inputs.push_back(*weights);
    }
    auto output = Variable::create(Expr::create(std::move(op), inputs));
    output->setName("y");
    auto buffer = Variable::save({output});

    Module::Config config;
    config.rearrange = true;
    config.shapeMutable = true;
    return std::shared_ptr<Module>(Module::load(shape.routeWeight ? std::vector<std::string>{"x", "ids", "w"}
                                                                  : std::vector<std::string>{"x", "ids"},
                                                {"y"}, reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                                indexedRuntime(), &config));
}

struct IndexedCase {
    IndexedShape shape;
    VARP input;
    VARP ids;
    VARP weights;
    std::shared_ptr<Module> module;
};

IndexedCase indexedCase(const IndexedShape& shape) {
    IndexedCase result{shape, nullptr, nullptr, nullptr, indexedModule(shape, &result.input, &result.ids, &result.weights)};
    if (!result.module) {
        MNN_ERROR("IndexedLinear speed module build failed for %s\n", shape.tag);
    }
    return result;
}

double indexedBench(const IndexedCase& test, int loop, int round) {
    std::vector<VARP> inputs = {test.input, test.ids};
    if (test.shape.routeWeight) {
        inputs.push_back(test.weights);
    }
    if (round < 0) {
        for (int i = 0; i < 3; ++i) {
            auto outputs = test.module->onForward(inputs);
            outputs.back()->readMap<float>();
            outputs.back()->unMap();
        }
        return 0.0;
    }

    Timer timer;
    std::vector<std::vector<VARP>> kept;
    kept.reserve(loop);
    for (int i = 0; i < loop; ++i) {
        kept.emplace_back(test.module->onForward(inputs));
    }
    kept.back().back()->readMap<float>();
    const double ms = static_cast<double>(timer.durationInUs()) / 1000.0 / loop;
    kept.back().back()->unMap();
    return ms;
}

}

class IndexedLinearSpeedTest : public MNNTestCase {
public:
    bool run(int precision) override {
        if (precision == 1) {
            MNN_PRINT("IndexedLinearSpeed: run with precision 2 (Low/fp16)\n");
            return true;
        }
        const int bits = indexedBits();
        const int block = 64;
        const int tokens = indexedTokens();
        std::vector<IndexedShape> shapes = {
            {"qwen_gup_decode", 1, 8, 2048, 1024, 64, 0, bits, block, true, 0},
            {"qwen_gup_prefill", tokens, 8, 2048, 1024, 64, 0, bits, block, true, 0},
            {"qwen_down_prefill", tokens, 8, 1024, 2048, 64, 1, bits, block, false, 0},
            {"q36_gup_decode", 1, 8, 2048, 1024, 256, 0, bits, block, true, 0},
            {"q36_gup_prefill", tokens, 8, 2048, 1024, 256, 0, bits, block, true, 0},
            {"q36_down_prefill", tokens, 8, 512, 2048, 256, 1, bits, block, false, 0},
            {"q36_down_rw", tokens, 8, 512, 2048, 256, 1, bits, block, false, 1},
            {"q36_down_rwraw", tokens, 8, 512, 2048, 256, 1, bits, block, false, 2},
            {"q36_down_rw3", tokens, 8, 512, 2048, 256, 1, bits, block, false, 3},
            {"mmoe_gup_decode", 1, 4, 768, 768, 60, 0, bits, block, true, 0},
            {"mmoe_gup_prefill", tokens, 4, 768, 768, 60, 0, bits, block, true, 0},
            {"mmoe_down_prefill", tokens, 4, 384, 768, 60, 1, bits, block, false, 0},
            {"mmoe_down_rwraw", tokens, 4, 384, 768, 60, 1, bits, block, false, 2},
        };
        if (bits != 0) {
            shapes.push_back({"mixtral_gup_decode", 1, 2, 4096, 28672, 8, 0, bits, block, true, 0});
            shapes.push_back({"mixtral_gup_prefill", tokens, 2, 4096, 28672, 8, 0, bits, block, true, 0});
            shapes.push_back({"mixtral_down_prefill", tokens, 2, 14336, 4096, 8, 1, bits, block, false, 0});
        }

        std::vector<IndexedCase> tests;
        tests.reserve(shapes.size());
        MNN_PRINT("--- build + warmup pass (untimed) ---\n");
        for (const auto& shape : shapes) {
            tests.emplace_back(indexedCase(shape));
            if (tests.back().module) {
                indexedBench(tests.back(), 1, -1);
            }
        }

        const int rounds = 3;
        for (const auto& test : tests) {
            if (!test.module) {
                continue;
            }
            const auto& shape = test.shape;
            const int loop = shape.tokens == 1 ? 100 : 5;
            double best = 1e30;
            for (int round = 0; round < rounds; ++round) {
                best = std::min(best, indexedBench(test, loop, round));
            }
            const int statusPrecision = MNNTestSuite::get()->pStaus.precision;
            const double elementBytes = shape.bits ? shape.bits / 8.0 : (statusPrecision == 1 ? 4.0 : 2.0);
            const int blocks = shape.bits ? shape.inputChannels / shape.blockChannels : 0;
            const int routes = shape.tokens * shape.topK;
            const double weights =
                routes * (static_cast<double>(shape.outputChannels) * shape.inputChannels * elementBytes +
                          (shape.bits ? static_cast<double>(shape.outputChannels) * blocks * 2 * sizeof(float) : 0.0));
            const double flops = 2.0 * routes * static_cast<double>(shape.outputChannels) * shape.inputChannels;
            MNN_PRINT(
                "BEST %-20s T=%-4d K=%d routes=%-5d E=%-3d I=%-5d O=%-5d bits=%d %8.3f ms %7.1f GB/s %7.1f GFLOPS\n",
                shape.tag, shape.tokens, shape.topK, routes, shape.experts, shape.inputChannels, shape.outputChannels,
                shape.bits, best, weights / (best * 1e6), flops / (best * 1e6));
        }
        return true;
    }
};
MNNTestSuiteRegister(IndexedLinearSpeedTest, "speed/IndexedLinear");

#endif
