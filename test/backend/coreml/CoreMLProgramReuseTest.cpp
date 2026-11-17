// Copyright © Alibaba Group Holding Limited. All rights reserved.
#include "MNNTestSuite.h"
#if defined(MNN_COREML_ENABLED) && MNN_COREML_ENABLED && defined(MNN_SUPPORT_TRANSFORMER_FUSE)
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include "MNN_generated.h"
#include "core/KVMeta.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
namespace {
using namespace MNN;
using namespace MNN::Express;
static bool check(VARP output, const std::vector<float>& reference, const char* label) {
    const float* data = output.get() ? output->readMap<float>() : nullptr;
    if (!data || output->getInfo()->size != reference.size()) {
        printf("FAIL %s missing output\n", label);
        return false;
    }
    double diff = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(data[i])) {
            printf("FAIL %s nonfinite\n", label);
            return false;
        }
        double d = data[i] - reference[i];
        diff += d * d;
        norm += reference[i] * reference[i];
        maximum = std::max(maximum, std::abs(d));
    }
    double rel = std::sqrt(diff / std::max(norm, 1e-20));
    printf("%s %s relative_l2=%.7g max_abs=%.7g\n", rel < 0.015 && maximum < 0.02 ? "PASS" : "FAIL", label, rel,
           maximum);
    fflush(stdout);
    return rel < 0.015 && maximum < 0.02;
}
static std::vector<float> values(int size, int seed) {
    std::vector<float> v(size);
    unsigned int state = seed;
    for (int i = 0; i < size; ++i) {
        state = state * 1664525u + 1013904223u;
        v[i] = 0.25f * ((state >> 8) / 16777216.0f - 0.5f);
    }
    return v;
}
static VARP input(const std::vector<int>& shape, const std::vector<float>& values, Dimensionformat order = NCHW) {
    auto x = _Input(shape, order);
    std::memcpy(x->writeMap<float>(), values.data(), values.size() * 4);
    return x;
}
static std::shared_ptr<Module> load(const std::vector<int8_t>& bytes, const std::vector<std::string>& names,
                                    const std::vector<std::string>& outputs,
                                    std::shared_ptr<Executor::RuntimeManager> rt) {
    Module::Config config;
    config.shapeMutable = true;
    config.rearrange = false;
    return std::shared_ptr<Module>(
        Module::load(names, outputs, (const uint8_t*)bytes.data(), bytes.size(), rt, &config), Module::destroy);
}
static std::vector<float> mm(const std::vector<float>& a, const std::vector<float>& b, int m, int k, int n) {
    std::vector<float> out(m * n, 0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int z = 0; z < k; ++z) {
                out[i * n + j] += a[i * k + z] * b[z * n + j];
            }
        }
    }
    return out;
}
class CoreMLProgramReuseTest : public MNNTestCase {
public:
    bool run(int) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_NN) {
            ++MNNTestSuite::get()->skipped;
            return true;
        }
        ScheduleConfig cfg;
        cfg.type = MNN_FORWARD_NN;
        cfg.numThread = 1;
        std::shared_ptr<Executor::RuntimeManager> rt(Executor::RuntimeManager::createRuntimeManager(cfg),
                                                     Executor::RuntimeManager::destroy);
        if (!rt || !rt->isBackendSupport({MNN_FORWARD_NN})[0]) {
            return false;
        }
        bool ok = true;
        {
            auto x = _Input({4, 64}, NCHW);
            x->setName("x");
            auto w = _Input({64, 64}, NCHW);
            w->setName("w");
            auto y = x;
            for (int layer = 0; layer < 4; ++layer) {
                y = _MatMul(y, w);
            }
            y->setName("y");
            auto model = load(Variable::save({y}), {"x", "w"}, {"y"}, rt);
            for (int m : {4, 1, 4}) {
                for (int seed : {7, 23}) {
                    auto a = values(m * 64, 1), b = values(64 * 64, seed), ref = a;
                    for (int i = 0; i < 4; ++i) {
                        ref = mm(ref, b, m, 64, 64);
                    }
                    auto out = model->onForward({input({m, 64}, a), input({64, 64}, b)});
                    ok = check(out.empty() ? nullptr : out[0], ref, "four_dynamic_matmuls") && ok;
                }
            }
        }
        {
            for (int modelSeed : {13, 29}) {
                auto x = _Input({3, 64, 1, 1}, NC4HW4);
                x->setName("x");
                auto y = x;
                std::vector<std::vector<float>> weights;
                std::vector<std::vector<float>> biases;
                for (int layer = 0; layer < 4; ++layer) {
                    weights.push_back(values(64 * 64, modelSeed + layer * 11));
                    biases.push_back(values(64, modelSeed + layer * 5));
                    y = _Conv(std::vector<float>(weights.back()), std::vector<float>(biases.back()), y, {64, 64},
                              {1, 1}, VALID);
                }
                y = _Convert(y, NCHW);
                y->setName("y");
                auto model = load(Variable::save({y}), {"x"}, {"y"}, rt);
                for (int m : {3, 1, 3}) {
                    auto a = values(m * 64, 2), ref = a;
                    for (int l = 0; l < 4; ++l) {
                        std::vector<float> trans(64 * 64);
                        for (int i = 0; i < 64; ++i) {
                            for (int j = 0; j < 64; ++j) {
                                trans[i * 64 + j] = weights[l][j * 64 + i];
                            }
                        }
                        ref = mm(ref, trans, m, 64, 64);
                        for (int i = 0; i < m; ++i) {
                            for (int j = 0; j < 64; ++j) {
                                ref[i * 64 + j] += biases[l][j];
                            }
                        }
                    }
                    auto out = model->onForward({_Convert(input({m, 64, 1, 1}, a), NC4HW4)});
                    ok = check(out.empty() ? nullptr : out[0], ref, "four_embedded_linears") && ok;
                }
            }
        }
        for (int variant = 0; variant < 3; ++variant) {
            bool depthwise = variant == 2;
            int kernel = variant == 0 ? 1 : 3;
            auto x = _Input({1, 3, 5, 5}, NC4HW4);
            x->setName("x");
            int channels = depthwise ? 3 : 5;
            auto y = _Conv(values(channels * (depthwise ? 1 : 3) * kernel * kernel, 31), values(channels, 17), x,
                           {3, channels}, {kernel, kernel}, VALID, {1, 1}, {1, 1}, depthwise ? 3 : 1, {0, 0}, false,
                           true);
            y = _Convert(y, NCHW);
            y->setName("y");
            auto bytes = Variable::save({y});
            ScheduleConfig cpu;
            cpu.type = MNN_FORWARD_CPU;
            std::shared_ptr<Executor::RuntimeManager> cpuRuntime(Executor::RuntimeManager::createRuntimeManager(cpu),
                                                                 Executor::RuntimeManager::destroy);
            auto referenceModel = load(bytes, {"x"}, {"y"}, cpuRuntime);
            auto model = load(bytes, {"x"}, {"y"}, rt);
            auto data = _Convert(input({1, 3, 5, 5}, values(75, 41)), NC4HW4);
            auto expected = referenceModel->onForward({data});
            auto actual = model->onForward({data});
            if (expected.empty() || actual.empty()) {
                return false;
            }
            auto ptr = expected[0]->readMap<float>();
            std::vector<float> reference(ptr, ptr + expected[0]->getInfo()->size);
            ok = check(actual[0], reference,
                       depthwise ? "depthwise_3x3_relu6" : (kernel == 1 ? "linear_spatial_relu6" : "conv_3x3_relu6")) && ok;
        }
        const char* experimental = std::getenv("MNN_COREML_EXPERIMENTAL_KV");
        if (!experimental || std::strcmp(experimental, "1") != 0) {
            MNN_PRINT("CoreML experimental KV tests disabled (MNN_COREML_EXPERIMENTAL_KV=1 to enable)\n");
            return ok;
        }
        for (bool packedValue : {false, true}) {
            int prefill = packedValue ? 18 : 4, heads = packedValue ? 14 : 4, dim = packedValue ? 64 : 16;
            KVMeta meta;
            rt->setHintPtr(Interpreter::KVCACHE_INFO, &meta);
            std::vector<VARP> ins, outs;
            std::vector<std::string> names, outNames;
            for (int layer = 0; layer < 2; ++layer) {
                std::vector<VARP> local;
                for (int i = 0; i < 3; ++i) {
                    auto x = i == 2 && packedValue
                                 ? _Input({prefill, 2 * dim, 1, 1}, NC4HW4)
                                 : _Input({1, prefill, i ? 2 : heads, dim}, packedValue ? NHWC : NCHW);
                    std::string n = "input" + std::to_string(layer * 3 + i);
                    x->setName(n);
                    ins.push_back(x);
                    names.push_back(n);
                    local.push_back(x);
                }
                auto mask = _Input(packedValue ? std::vector<int>{1, 1, prefill, prefill} : std::vector<int>{}, NCHW);
                mask->setName("mask" + std::to_string(layer));
                names.push_back(mask->name());
                ins.push_back(mask);
                local.push_back(mask);
                std::unique_ptr<OpT> op(new OpT);
                op->type = OpType_Attention;
                op->main.type = OpParameter_AttentionParam;
                op->main.value = new AttentionParamT;
                op->main.AsAttentionParam()->kv_cache = true;
                auto y = Variable::create(Expr::create(op.get(), local));
                auto n = "out" + std::to_string(layer);
                y->setName(n);
                outs.push_back(y);
                outNames.push_back(n);
            }
            auto model = load(Variable::save(outs), names, outNames, rt);
            std::shared_ptr<Module> decode;
            std::vector<float> keys[2], vals[2];
            for (int step = 0; step < 8; ++step) {
                int t = step == 0 || step == 5 ? prefill : 1;
                if (step == 5) {
                    meta.remove = meta.previous;
                    for (int l = 0; l < 2; ++l) {
                        keys[l].clear();
                        vals[l].clear();
                    }
                }
                meta.add = t;
                int past = meta.previous - meta.remove;
                std::vector<VARP> inputs;
                std::vector<std::vector<float>> refs;
                for (int l = 0; l < 2; ++l) {
                    auto q = values(t * heads * dim, step * 7 + l * 71), k = values(t * 2 * dim, step * 11 + l * 43),
                         v = values(t * 2 * dim, step * 13 + l * 97);
                    keys[l].insert(keys[l].end(), k.begin(), k.end());
                    vals[l].insert(vals[l].end(), v.begin(), v.end());
                    inputs.push_back(input({1, t, heads, dim}, q, packedValue ? NHWC : NCHW));
                    inputs.push_back(input({1, t, 2, dim}, k, packedValue ? NHWC : NCHW));
                    if (packedValue) {
                        inputs.push_back(_Convert(input({t, 2 * dim, 1, 1}, v), NC4HW4));
                        std::vector<float> mask(t * t, 0.0f);
                        for (int r = 0; r < t; ++r) {
                            for (int c = r + 1; c < t; ++c) {
                                mask[r * t + c] = -65504.0f;
                            }
                        }
                        inputs.push_back(input({1, 1, t, t}, mask));
                    } else {
                        inputs.push_back(input({1, t, 2, dim}, v));
                        inputs.push_back(input({}, {0}));
                    }
                    std::vector<float> ref(t * heads * dim, 0);
                    for (int row = 0; row < t; ++row) {
                        for (int h = 0; h < heads; ++h) {
                            int len = past + row + 1;
                            std::vector<float> score(len);
                            float mx = -1e20;
                            for (int col = 0; col < len; ++col) {
                                float s = 0;
                                for (int d = 0; d < dim; ++d) {
                                    s +=
                                        q[(row * heads + h) * dim + d] * keys[l][(col * 2 + h / (heads / 2)) * dim + d];
                                }
                                score[col] = s / std::sqrt(static_cast<float>(dim));
                                mx = std::max(mx, score[col]);
                            }
                            float sum = 0;
                            for (auto& s : score) {
                                s = std::exp(s - mx);
                                sum += s;
                            }
                            for (int d = 0; d < dim; ++d) {
                                for (int col = 0; col < len; ++col) {
                                    ref[(row * heads + h) * dim + d] +=
                                        score[col] / sum * vals[l][(col * 2 + h / (heads / 2)) * dim + d];
                                }
                            }
                        }
                    }
                    refs.push_back(ref);
                }
                auto out = (step > 0 && step < 5 ? decode : model)->onForward(inputs);
                if (step == 0) {
                    decode.reset(Module::clone(model.get()), Module::destroy);
                }
                for (int l = 0; l < 2; ++l) {
                    ok = check(out.size() > l ? out[l] : nullptr, refs[l], "independent_layer_kv") && ok;
                }
                meta.sync();
            }
        }
        printf("RESULT %s\n", ok ? "PASS" : "FAIL");
        return ok;
    }
};
} // namespace
MNNTestSuiteRegister(CoreMLProgramReuseTest, "backend/coreml/program_reuse");
#endif
