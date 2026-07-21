//
//  QKVFusionTest.cpp
//  MNNTests
//
//  Tests correctness of QKV single-dispatch fusion for Metal LLM decode.
//  Creates three quantized Conv1x1 ops sharing the same input (simulating
//  Q/K/V projections in a transformer attention layer) and verifies that
//  fused dispatch produces the same results as non-fused execution.
//
//  This test runs on any backend:
//    - CPU: no fusion, serves as baseline correctness check
//    - Metal: QKV fusion is automatically applied when conditions are met
//
//  Usage:
//    ./run_test.out op/QKVFusion 1 2    # Metal backend, precision Low
//    ./run_test.out op/QKVFusion 0 1    # CPU backend (baseline)
//

#include <math.h>
#include <vector>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "MNNTestSuite.h"
#include "CommonOpCreator.hpp"

using namespace MNN::Express;
using namespace MNN;

class QKVFusionTest : public MNNTestCase {
public:
    virtual ~QKVFusionTest() = default;

    // Create a single quantized Conv1x1 and return the output VARP
    // along with the reference FP32 conv output for comparison.
    static std::pair<VARP, VARP> makeQuantConv1x1(
        VARP x, int ic, int oc, int nbit, int blocksize,
        std::vector<float>& weightStorage, std::vector<float>& biasStorage,
        std::vector<float>& scaleStorage, std::vector<float>& newWeightStorage) {

        float fac = 0.23f;
        int res = 10;
        float tail = 0.05f;
        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold - 1;
        int8_t xMin = -(1 << (nbit - 1));

        int blocknum = 1;
        if (blocksize == 0 || ic % blocksize != 0) {
            blocksize = ic;
            blocknum = 1;
        } else {
            blocknum = ic / blocksize;
        }

        weightStorage.resize(oc * ic);
        biasStorage.resize(oc);
        scaleStorage.resize(2 * oc * blocknum);
        newWeightStorage.resize(oc * ic);

        // Initialize weight and bias with deterministic patterns
        for (int i = 0; i < oc; ++i) {
            biasStorage[i] = (i % 10) * 0.005f;
            for (int j = 0; j < ic; ++j) {
                weightStorage[i * ic + j] = ((i * ic + j) % res) * fac + tail;
            }
        }

        // Quantize weights and compute scale/offset
        for (int k = 0; k < oc; ++k) {
            for (int b = 0; b < blocknum; ++b) {
                auto index = k * blocknum + b;
                auto minmax = MNN::findMinMax(
                    weightStorage.data() + k * ic + b * blocksize,
                    blocksize);
                auto scale_ = (minmax.second - minmax.first) / (threshold - clampMin);
                scaleStorage[2 * index] = minmax.first;
                scaleStorage[2 * index + 1] = scale_;

                for (int u = 0; u < blocksize; ++u) {
                    int idx = k * ic + b * blocksize + u;
                    int q_weight = (int)((weightStorage[idx] - minmax.first) *
                                         (threshold - clampMin) / (minmax.second - minmax.first) + clampMin);
                    newWeightStorage[idx] = (q_weight - xMin) * scale_ + minmax.first;
                }
            }
        }

        // Create quantized conv
        auto biasQ = biasStorage;
        auto scaleQ = scaleStorage;
        auto quantConv = _HybridConv(weightStorage, std::move(biasQ), std::move(scaleQ),
                                      x, {ic, oc}, {1, 1}, PaddingMode::CAFFE,
                                      {1, 1}, {1, 1}, 1, {0, 0},
                                      false, false, nbit, true);

        // Create FP32 reference conv
        auto biasFp = biasStorage;
        auto wFp = newWeightStorage;
        auto refConv = _Conv(std::move(wFp), std::move(biasFp), x,
                             {ic, oc}, {1, 1}, PaddingMode::CAFFE,
                             {1, 1}, {1, 1}, 1, {0, 0});

        return {quantConv, refConv};
    }

    // Test QKV fusion with given dimensions and quantization bits
    static bool testQKVFusion(int ic, int oc_q, int oc_kv, int nbit, int blocksize, int precision) {
        // Input: batch=1, ic channels, 1x1 spatial (decode mode)
        VARP x = _Input({1, ic, 1, 1}, NCHW, halide_type_of<float>());
        auto xPtr = x->writeMap<float>();
        int8_t xMax = (1 << (nbit - 1)) - 1;
        for (int i = 0; i < ic; ++i) {
            xPtr[i] = (i % (2 * xMax + 1) - xMax) * 0.017f;
        }
        x = _Convert(x, NC4HW4);

        // Create Q, K, V projections sharing the same input
        std::vector<float> wQ, bQ, sQ, nwQ;
        std::vector<float> wK, bK, sK, nwK;
        std::vector<float> wV, bV, sV, nwV;

        auto qPair = makeQuantConv1x1(x, ic, oc_q, nbit, blocksize, wQ, bQ, sQ, nwQ);
        auto kPair = makeQuantConv1x1(x, ic, oc_kv, nbit, blocksize, wK, bK, sK, nwK);
        auto vPair = makeQuantConv1x1(x, ic, oc_kv, nbit, blocksize, wV, bV, sV, nwV);
        VARP qOut = qPair.first, qRef = qPair.second;
        VARP kOut = kPair.first, kRef = kPair.second;
        VARP vOut = vPair.first, vRef = vPair.second;

        // Convert to NCHW for comparison
        qOut = _Convert(qOut, NCHW);
        kOut = _Convert(kOut, NCHW);
        vOut = _Convert(vOut, NCHW);
        qRef = _Convert(qRef, NCHW);
        kRef = _Convert(kRef, NCHW);
        vRef = _Convert(vRef, NCHW);

        // Read results
        auto qOutPtr = qOut->readMap<float>();
        auto kOutPtr = kOut->readMap<float>();
        auto vOutPtr = vOut->readMap<float>();
        auto qRefPtr = qRef->readMap<float>();
        auto kRefPtr = kRef->readMap<float>();
        auto vRefPtr = vRef->readMap<float>();

        if (!qOutPtr || !kOutPtr || !vOutPtr || !qRefPtr || !kRefPtr || !vRefPtr) {
            MNN_ERROR("QKVFusion: failed to read output maps\n");
            return false;
        }

        // Compare quantized output vs FP32 reference
        // Allow reasonable tolerance for quantization error
        float relTol = 0.1f;  // 10% relative tolerance for quantized vs fp32
        float absTol = 0.5f;  // absolute tolerance

        auto checkOutput = [&](const float* out, const float* ref, int size, const char* name) -> bool {
            float maxErr = 0.0f;
            float maxVal = 0.001f;
            for (int i = 0; i < size; ++i) {
                maxVal = fmaxf(maxVal, fabsf(ref[i]));
            }
            for (int i = 0; i < size; ++i) {
                float err = fabsf(out[i] - ref[i]);
                maxErr = fmaxf(maxErr, err);
                if (err > absTol && err / maxVal > relTol) {
                    MNN_ERROR("QKVFusion %s[%d]: got %.6f, expected %.6f, err=%.6f (maxVal=%.6f)\n",
                              name, i, out[i], ref[i], err, maxVal);
                    return false;
                }
            }
            return true;
        };

        bool ok = true;
        ok = ok && checkOutput(qOutPtr, qRefPtr, oc_q, "Q");
        ok = ok && checkOutput(kOutPtr, kRefPtr, oc_kv, "K");
        ok = ok && checkOutput(vOutPtr, vRefPtr, oc_kv, "V");

        if (!ok) {
            MNN_ERROR("QKVFusion test FAILED: ic=%d, oc_q=%d, oc_kv=%d, nbit=%d, blocksize=%d\n",
                      ic, oc_q, oc_kv, nbit, blocksize);
        }
        return ok;
    }

    virtual bool run(int precision) override {
        bool allPass = true;

        // Test 1: 4-bit quantization, typical GQA shape (Q=4096, K=V=1024)
        // Using smaller sizes to keep test fast but still trigger QKV fusion path
        {
            // Small shape: ic=128, Q_oc=64, KV_oc=16, blocksize=32
            bool ok = testQKVFusion(128, 64, 16, 4, 32, precision);
            if (!ok) {
                MNN_ERROR("QKVFusion W4 small GQA test FAILED\n");
                allPass = false;
            }
        }

        // Test 2: 8-bit quantization, same GQA shape
        // This specifically tests the W_QUANT_8 weight pointer bug fix
        {
            bool ok = testQKVFusion(128, 64, 16, 8, 32, precision);
            if (!ok) {
                MNN_ERROR("QKVFusion W8 small GQA test FAILED\n");
                allPass = false;
            }
        }

        // Test 3: 4-bit quantization, MHA shape (Q=K=V same output channels)
        {
            bool ok = testQKVFusion(128, 32, 32, 4, 32, precision);
            if (!ok) {
                MNN_ERROR("QKVFusion W4 MHA test FAILED\n");
                allPass = false;
            }
        }

        // Test 4: 8-bit quantization, MHA shape
        {
            bool ok = testQKVFusion(128, 32, 32, 8, 32, precision);
            if (!ok) {
                MNN_ERROR("QKVFusion W8 MHA test FAILED\n");
                allPass = false;
            }
        }

        // Test 5: 4-bit quantization, larger shape closer to real models
        // ic=256, Q_oc=256, KV_oc=64 (GQA ratio = 4)
        {
            bool ok = testQKVFusion(256, 256, 64, 4, 64, precision);
            if (!ok) {
                MNN_ERROR("QKVFusion W4 large GQA test FAILED\n");
                allPass = false;
            }
        }

        // Test 6: 8-bit quantization, larger shape
        {
            bool ok = testQKVFusion(256, 256, 64, 8, 64, precision);
            if (!ok) {
                MNN_ERROR("QKVFusion W8 large GQA test FAILED\n");
                allPass = false;
            }
        }

        return allPass;
    }
};

MNNTestSuiteRegister(QKVFusionTest, "op/QKVFusion");