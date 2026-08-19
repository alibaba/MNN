//
//  MetalConvWQuantTest.cpp
//  MNNTests
//
//  Numerical-correctness test for the Metal low-memory (weight-quantized) 1x1
//  convolution across W_QUANT_2 / 3 / 4 / 8. For each (shape, bits) the test
//  quantizes + dequantizes the weights to the n-bit grid, computes an fp32
//  reference conv1x1 with those dequantized weights, runs the same op on the
//  Metal backend, and compares. Because the reference uses the *dequantized*
//  weights, the comparison is backend-vs-backend (not quantization-limited), so
//  a tight relative-error threshold catches real kernel bugs.
//
//  Kernel coverage (single conv op, Apple-GPU with simdgroup-matrix, e.g. M4):
//    - conv1x1_gemv_g4m1_2sg_wquant_sg   decode area==1, +/- SPLIT_K_2
//    - conv1x1_gemv_g16_wquant_sg        lm_head (oc > 16384)
//    - conv1x1_gemv_g4mN_wquant_sg       multi-token W4/8 (in-shader dequant)
//    - conv1x1_gemm_*_wquant_sg          prefill sg-matrix gemm (W4/8)
//    - conv1x1_w_dequant + fp gemm       outer-dequant path (W2/3 prefill)
//    - conv1x1_fused_q4_gemm_stage[_m64] Q4 prefill on tensor-API devices
//  NOT reachable from a single op on such a device (documented, not covered):
//    - conv1x1_gemv_g8_wquant_sg W2/3 multi-token: only selected when the device
//      lacks simdgroup-matrix (e.g. A13); M-series routes W2/3 prefill to the
//      outer-dequant path instead (see MetalConvolution1x1.mm).
//    - fusion variants (GATE_UP_FUSED / QKV_FUSED / LN_FUSED / ROW_2): only built
//      by fusion leaders inside a full LLM graph, not by an isolated conv op.
//
//  Requires the low-memory weight-quant path (MNN_LOW_MEMORY); skipped otherwise.
//
#ifdef MNN_LOW_MEMORY

#include <cmath>
#include <cstring>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "CommonOpCreator.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace {

// Quantize + dequantize weights in place, per block, mirroring the proven
// round-trip logic in ConvolutionTest.cpp (ConvolutionInt8CommonTest). After
// this call `weight` holds dequantized floats and `alpha` the per-block
// scale/bias that _HybridConv -> IDSTEncoder re-quantizes losslessly.
// weight layout: [oc][ic] row-major. async=true -> asymmetric (min,scale)/block,
// alpha size 2*oc*blockNum; async=false -> symmetric scale/block, alpha size
// oc*blockNum.
void quantizeDequantize(std::vector<float>& weight, std::vector<float>& alpha, int ic, int oc, int blockSize, int nbit,
                        bool async) {
    int blockNum = ic / blockSize;
    float threshold = (float)(1 << (nbit - 1)) - 1.0f;
    float clampMin = -threshold;
    if (async) {
        clampMin = -threshold - 1;
    }
    alpha.resize(async ? 2 * oc * blockNum : oc * blockNum);
    for (int o = 0; o < oc; ++o) {
        for (int b = 0; b < blockNum; ++b) {
            int begin = o * ic + b * blockSize;
            if (async) {
                float minValue = weight[begin], maxValue = weight[begin];
                for (int i = 1; i < blockSize; ++i) {
                    minValue = fmin(minValue, weight[begin + i]);
                    maxValue = fmax(maxValue, weight[begin + i]);
                }
                float range = maxValue - minValue;
                float scale = 0.f;
                if (range >= 1e-6f) {
                    scale = range / (threshold - clampMin);
                }
                alpha[2 * (o * blockNum + b)] = minValue;
                alpha[2 * (o * blockNum + b) + 1] = scale;
                float inv = (scale >= 1e-6f) ? (1.0f / scale) : 0.0f;
                for (int i = 0; i < blockSize; ++i) {
                    float* p = &weight[begin + i];
                    int code = (int)std::round((*p - minValue) * inv + clampMin);
                    code = (int)fmax(fmin((float)code, threshold), clampMin);
                    *p = ((float)code - clampMin) * scale + minValue;
                }
            } else {
                float absMax = 1e-8f;
                for (int i = 0; i < blockSize; ++i) {
                    absMax = fmax(absMax, fabs(weight[begin + i]));
                }
                float scale = absMax / threshold;
                alpha[o * blockNum + b] = scale;
                float inv = (scale >= 1e-6f) ? (1.0f / scale) : 0.0f;
                for (int i = 0; i < blockSize; ++i) {
                    float* p = &weight[begin + i];
                    int code = (int)fmax(fmin(round(*p * inv), threshold), clampMin);
                    *p = (float)code * scale;
                }
            }
        }
    }
}

// fp32 reference for 1x1 stride-1 pad-0 group-1 conv: out[o][a] = bias[o] +
// sum_c Wdeq[o][c] * in[c][a]. Input NCHW {1, ic, ih, iw}; area = ih*iw.
void referenceConv1x1(const std::vector<float>& in, const std::vector<float>& weight, const std::vector<float>& bias,
                      std::vector<float>& out, int ic, int oc, int area, bool relu, bool relu6) {
    out.assign((size_t)oc * area, 0.f);
    for (int o = 0; o < oc; ++o) {
        const float* wRow = weight.data() + (size_t)o * ic;
        for (int c = 0; c < ic; ++c) {
            float w = wRow[c];
            const float* inRow = in.data() + (size_t)c * area;
            for (int a = 0; a < area; ++a) {
                out[(size_t)o * area + a] += w * inRow[a];
            }
        }
        for (int a = 0; a < area; ++a) {
            float v = out[(size_t)o * area + a] + bias[o];
            if (relu6) {
                v = fmin(6.f, fmax(0.f, v)); // ReLU6 = clamp to [0, 6]
            } else if (relu) {
                v = fmax(0.f, v);
            }
            out[(size_t)o * area + a] = v;
        }
    }
}

struct CaseShape {
    int ic, oc, ih, iw, blockSize;
    const char* kernelNote; // which compute shader this shape is meant to hit
    bool q4Only;
};

} // namespace

class MetalConvWQuantTest : public MNNTestCase {
public:
    virtual ~MetalConvWQuantTest() = default;

    bool testUnit(MNNForwardType type, int ic, int oc, int ih, int iw, int blockSize, int nbit, bool async,
                  int precision, const char* kernelNote) {
        int area = ih * iw;
        // Deterministic pseudo-random weights in [-0.5, 0.5).
        std::vector<float> weight((size_t)oc * ic);
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = ((float)((i * 1103515245u + 12345u) % 65536) / 65536.0f) - 0.5f;
        }
        std::vector<float> bias(oc);
        for (int o = 0; o < oc; ++o) {
            bias[o] = ((float)((o * 2654435761u) % 65536) / 65536.0f) - 0.5f;
        }
        std::vector<float> input((size_t)ic * area);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = ((float)((i * 40503u) % 65536) / 65536.0f) - 0.5f;
        }

        std::vector<float> alpha;
        quantizeDequantize(weight, alpha, ic, oc, blockSize, nbit, async);
        std::vector<float> ref;

        auto x = _Input({1, ic, ih, iw}, NCHW, halide_type_of<float>());
        ::memcpy(x->writeMap<float>(), input.data(), input.size() * sizeof(float));
        x->unMap();

        // relu/relu6 exercised separately on the no-activation reference.
        const bool activations[3][2] = {{false, false}, {true, false}, {false, true}};
        for (int act = 0; act < 3; ++act) {
            bool relu = activations[act][0], relu6 = activations[act][1];
            referenceConv1x1(input, weight, bias, ref, ic, oc, area, relu, relu6);
            auto y = _HybridConv(weight, bias, alpha, x, {ic, oc}, {1, 1}, CAFFE, {1, 1}, {1, 1}, 1, {0, 0}, relu,
                                 relu6, nbit, async);
            y = _Convert(y, NCHW);
            auto ptr = y->readMap<float>();
            if (ptr == nullptr) {
                MNN_ERROR("MetalConvWQuant readMap null (ic=%d oc=%d area=%d w%d async=%d act=%d)\n", ic, oc, area,
                          nbit, async, act);
                return false;
            }
            if (!checkVectorByRelativeError<float>(ptr, ref.data(), (int)ref.size(), 0.005f)) {
                MNN_ERROR("MetalConvWQuant FAILED: %s ic=%d oc=%d area=%d w%d async=%d relu=%d relu6=%d\n", kernelNote,
                          ic, oc, area, nbit, async, relu, relu6);
                return false;
            }
        }
        return true;
    }

    virtual bool run(int precision) override {
        auto status = MNNTestSuite::get()->pStaus;
        MNNForwardType type = (MNNForwardType)status.forwardType;
        // Metal-specific correctness test. On a non-Metal backend there is nothing to
        // validate here (and the CPU low-memory W2/W3 path is not reliable), so skip
        // gracefully instead of failing. Run with backend arg 1 (MNN_FORWARD_METAL).
        if (type != MNN_FORWARD_METAL) {
            MNN_PRINT("MetalConvWQuant: skipped (Metal-only test, backend=%d). Run with backend=1.\n", (int)type);
            return true;
        }
        BackendConfig bnConfig;
        bnConfig.precision = (BackendConfig::PrecisionMode)precision;
        bnConfig.memory = BackendConfig::Memory_Low; // required to select the low-memory quant conv path
        auto exe = Executor::newExecutor(type, bnConfig, 1);
        ExecutorScope scope(exe);

        const char* backendName = "Metal";
        MNN_PRINT("\n## MetalConvWQuant (backend=%s, precision=%d)\n", backendName, precision);

        // Shape matrix -> compute shader (see file header). area = ih*iw. The two
        // in-shader shapes use oc=2176 so ic*oc strictly exceeds the 4M threshold that
        // keeps W4/8 in the in-shader dequant path (g4mN / sg-matrix gemm).
        std::vector<CaseShape> shapes = {
            {512, 256, 1, 1, 32, "2sg decode + SPLIT_K_2 (oc%8==0)", false},
            {512, 252, 1, 1, 32, "2sg decode no-splitk (oc%8!=0)", false},
            {2048, 64, 1, 1, 32, "2sg decode + Q4 block32 W16", true},
            {2048, 64, 1, 1, 64, "2sg decode + Q4 block64 W16", true},
            {2048, 64, 1, 1, 128, "2sg decode + Q4 block128 W16", true},
            {2048, 64, 1, 1, 256, "2sg decode + Q4 block256 W16", true},
            {6144, 64, 1, 1, 64, "2sg decode + Q4 block64 W16 (96 blocks)", true},
            {128, 16400, 1, 1, 32, "g16 lm_head (oc>16384)", false},
            {2048, 16400, 1, 1, 64, "g16 lm_head + Q4 block64 W16", true},
            {512, 16400, 1, 1, 128, "g16 lm_head + Q4 block128 W16", true},
            {512, 16400, 1, 1, 256, "g16 lm_head + Q4 block256 W16", true},
            {2048, 64, 8, 8, 32, "fused Q4 GEMM M32 + block32 fp16 metadata", true},
            {2048, 64, 8, 8, 64, "fused Q4 GEMM M32 + block64 fp16 metadata", true},
            {2048, 64, 8, 8, 128, "fused Q4 GEMM M32 + block128 fp16 metadata", true},
            {2048, 64, 8, 8, 256, "fused Q4 GEMM M32 + block256 fp16 metadata", true},
            {6144, 64, 8, 8, 64, "fused Q4 GEMM M32 + block64 fp16 metadata (96 blocks)", true},
            {2048, 64, 8, 16, 32, "fused Q4 GEMM M64 + block32 fp16 metadata", true},
            {2048, 64, 8, 16, 64, "fused Q4 GEMM M64 + block64 fp16 metadata", true},
            {2048, 64, 8, 16, 128, "fused Q4 GEMM M64 + block128 fp16 metadata", true},
            {2048, 64, 8, 16, 256, "fused Q4 GEMM M64 + block256 fp16 metadata", true},
            {6144, 64, 8, 16, 64, "fused Q4 GEMM M64 + block64 fp16 metadata (96 blocks)", true},
            {256, 128, 8, 8, 32, "outer-dequant pre-pass + fp gemm (all bits; area=64, ic*oc<4M)", false},
            {2048, 2176, 1, 2, 32, "g4mN multi-token in-shader (W4/8)", false},
            {2048, 2176, 1, 2, 64, "g4mN multi-token + Q4 block64", true},
            {2048, 2176, 4, 8, 32, "gemm sg-matrix prefill (W4/8)", false},
            {2048, 2176, 4, 8, 64, "gemm sg-matrix prefill + Q4 block64", true},
        };

        bool allPass = true;
        for (auto& s : shapes) {
            // g4mN / in-shader gemm kernels have no true W2/3 branches: when area>1 and
            // ic*oc is large enough to keep W4/8 in-shader, W2/3 is routed to the
            // outer-dequant path instead (already covered by the area=64 shape above), so
            // skip W2/3 for exactly those two shapes.
            bool skipW23 = (s.ih * s.iw > 1) && ((size_t)s.ic * s.oc > (size_t)4 * 1024 * 1024);
            for (int nbit : {2, 3, 4, 8}) {
                if (s.q4Only && nbit != 4) {
                    continue;
                }
                if ((nbit == 2 || nbit == 3) && skipW23) {
                    continue;
                }
                for (bool async : {false, true}) {
                    bool ok = testUnit(type, s.ic, s.oc, s.ih, s.iw, s.blockSize, nbit, async, precision, s.kernelNote);
                    if (!ok) {
                        allPass = false;
                    }
                }
            }
        }
        if (allPass) {
            MNN_PRINT("MetalConvWQuant all cases passed.\n");
        }
        return allPass;
    }
};

MNNTestSuiteRegister(MetalConvWQuantTest, "op/conv_wquant_metal");

#endif // MNN_LOW_MEMORY
