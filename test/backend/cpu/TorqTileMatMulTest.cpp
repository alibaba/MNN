// Copyright © 2026, Alibaba Group Holding Limited
#if defined(MNN_BUILD_STATIC_LIBS) && defined(MNN_USE_TORQ_TILE)
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/riscv/rvv/torq-tile/MNNTorqTileFunctions.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

class TorqTileMatMulTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto core = MNN::MNNGetCoreFunctions();
        if (!core->supportRVV) {
            MNN_PRINT("TORQ-Tile requires an RVV-capable CPU\n");
            return false;
        }
        if (core->MNNPackedMatMul != MNNPackedMatMulFP32_TorqTile ||
            core->MNNPackedMatMulRemain != MNNPackedMatMulRemainFP32_TorqTile) {
            MNN_ERROR("TORQ-Tile was built but was not registered\n");
            return false;
        }
        int eP, lP, hP;
        core->MNNGetMatMulPackMode(&eP, &lP, &hP);
        if (eP != 16 || lP != 1 || hP != 4) {
            return false;
        }
        const int depths[] = {0, 1, 3, 7, 16, 33};
        const int channels[] = {0, 1, 3, 4, 5, 8, 17};
        size_t cases = 0;
        for (int e = 0; e <= eP; ++e) {
            for (int depth : depths) {
                for (int h : channels) {
                    for (bool bias : {false, true}) {
                        for (int post = 0; post < 3; ++post) {
                            for (int extra : {0, 3}) {
                                if (!check(e, depth, h, bias, post, eP + extra, extra, 1, false)) {
                                    return false;
                                }
                                ++cases;
                            }
                        }
                    }
                }
            }
        }
        // Full-tile ABI ignores parameter[0]; threaded slices must honor it.
        for (int threads : {1, 4}) {
            for (int depth : depths) {
                if (!check(eP, depth, 17, true, 2, eP, 3, threads, threads == 1)) {
                    return false;
                }
                ++cases;
            }
        }
        if (!checkSpecialValues()) {
            return false;
        }
        // Capability and incompatible-layout gates must leave the table alone.
        auto copy = *core;
        copy.supportRVV = false;
        copy.MNNPackedMatMul = nullptr;
        MNNTorqTileInitializeFunctions(&copy);
        if (copy.MNNPackedMatMul != nullptr) {
            return false;
        }
        copy.supportRVV = true;
        copy.bytes = 2;
        MNNTorqTileInitializeFunctions(&copy);
        if (copy.MNNPackedMatMul != nullptr) {
            return false;
        }
        MNN_PRINT("TORQ-Tile packed MatMul: %zu cases passed\n", cases);
        return true;
    }

private:
    static bool checkSpecialValues() {
        auto core = MNN::MNNGetCoreFunctions();
        const float values[] = {0.0f,
                                -0.0f,
                                std::numeric_limits<float>::denorm_min(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()};
        const float weights[] = {0.5f, -0.5f, 0.0f, 1.0f};
        for (int e : {1, 3, 16}) {
            for (float value : values) {
                for (bool clamp : {false, true}) {
                    std::vector<float> a(16, value), output(e * 4);
                    size_t parameters[] = {16 * sizeof(float), 1, 4, size_t(e * 4 * sizeof(float)), 0, 0};
                    float post[] = {1, 1, -0.5f, 0.75f};
                    core->MNNPackedMatMulRemain(output.data(), a.data(), weights, e, parameters, clamp ? post : nullptr,
                                                nullptr, nullptr, nullptr);
                    for (int x = 0; x < e * 4; ++x) {
                        float expected = std::fma(value, weights[x % 4], 0.0f);
                        if (clamp) {
                            // E=1 uses the original scalar path. Other rows
                            // follow RVV's numeric-operand min/max semantics.
                            expected = e == 1 ? std::min(std::max(expected, post[2]), post[3])
                                              : std::fmin(std::fmax(expected, post[2]), post[3]);
                        }
                        if (!(std::isnan(expected) ? std::isnan(output[x]) : output[x] == expected)) {
                            MNN_ERROR("TORQ-Tile special-value mismatch E=%d clamp=%d index=%d\n", e, clamp, x);
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    static bool check(int e, int depth, int h, bool biasOn, int post, int aStride, int extra, int threads, bool full) {
        const float guard = 12345.0f;
        const int h4 = (h + 3) / 4;
        const int bStride = depth * 4 + extra;
        const int cStride = e * 4 + 8;
        std::vector<float> a(depth * aStride + 8, guard), b(h4 * bStride + 8, guard);
        std::vector<float> bias(h4 * 4 + 8, guard), actual(h4 * cStride + 8, guard);
        std::vector<float> expected = actual;
        for (int z = 0; z < depth; ++z) {
            for (int x = 0; x < e; ++x) {
                a[4 + z * aStride + x] = ((z * 7 + x * 3) % 17 - 8) * 0.125f;
            }
        }
        for (int y = 0; y < h4; ++y) {
            for (int lane = 0; lane < 4; ++lane) {
                bias[4 + y * 4 + lane] = ((y * 4 + lane) % 5 - 2) * 0.25f;
                for (int z = 0; z < depth; ++z) {
                    b[4 + y * bStride + z * 4 + lane] = ((z * 5 + y * 3 + lane) % 13 - 6) * 0.0625f;
                }
                for (int x = 0; x < e; ++x) {
                    double value = biasOn ? bias[4 + y * 4 + lane] : 0.0;
                    for (int z = 0; z < depth; ++z) {
                        value += double(a[4 + z * aStride + x]) * b[4 + y * bStride + z * 4 + lane];
                    }
                    if (post == 2) {
                        value = std::fmin(std::fmax(value, -0.5), 0.75);
                    }
                    expected[4 + y * cStride + x * 4 + lane] = float(value);
                }
            }
        }
        auto aBefore = a, bBefore = b, biasBefore = bias;
        size_t parameters[] = {
            size_t(full ? 4 : aStride * 4), size_t(depth), size_t(h), size_t(cStride * 4), 0, size_t(extra * 4)};
        float postParameters[] = {1, 1, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        if (post == 2) {
            postParameters[2] = -0.5f;
            postParameters[3] = 0.75f;
        }
        auto postPtr = post ? postParameters : nullptr;
        auto biasPtr = biasOn ? bias.data() + 4 : nullptr;
        auto core = MNN::MNNGetCoreFunctions();
        if (full) {
            core->MNNPackedMatMul(actual.data() + 4, a.data() + 4, b.data() + 4, parameters, postPtr, biasPtr, nullptr,
                                  nullptr);
        } else if (threads == 1) {
            core->MNNPackedMatMulRemain(actual.data() + 4, a.data() + 4, b.data() + 4, e, parameters, postPtr, biasPtr,
                                        nullptr, nullptr);
        } else {
            std::vector<std::thread> workers;
            for (int t = 0; t < threads; ++t) {
                workers.emplace_back([&, t]() {
                    const int start = t * e / threads;
                    const int end = (t + 1) * e / threads;
                    core->MNNPackedMatMulRemain(actual.data() + 4 + start * 4, a.data() + 4 + start, b.data() + 4,
                                                end - start, parameters, postPtr, biasPtr, nullptr, nullptr);
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            if ((expected[i] == guard && actual[i] != guard) || !std::isfinite(actual[i]) ||
                std::abs(actual[i] - expected[i]) > 1e-5f * (1 + std::abs(expected[i]))) {
                MNN_ERROR("TORQ-Tile mismatch E=%d K=%d H=%d bias=%d post=%d extra=%d threads=%d index=%zu\n", e, depth,
                          h, biasOn, post, extra, threads, i);
                return false;
            }
        }
        return a == aBefore && b == bBefore && bias == biasBefore;
    }
};

MNNTestSuiteRegister(TorqTileMatMulTest, "op/torq_tile_matmul");
#endif
