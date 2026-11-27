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
//  Kernel coverage (single conv op, Apple-GPU with simdgroup-matrix):
//    - conv1x1_gemv_g4m1_2sg_wquant_sg   decode area==1, +/- split K
//    - conv1x1_gemv_g16_wquant_sg        lm_head (oc > 16384)
//    - conv1x1_gemv_g4mN_wquant_sg       multi-token W4/8 (in-shader dequant)
//    - conv1x1_gemm_*_wquant_sg          prefill sg-matrix gemm (W4/8)
//    - conv1x1_w_dequant + fp gemm       outer-dequant path (W2/3 prefill)
//    - conv1x1_fused_q4_gemm_stage[_m64] Q4 prefill on tensor-API devices
//    - conv1x1_fused_q8_gemm_stage_m8    W8 small-area prefill on tensor-API devices
//  NOT reachable from a single op on such a device (documented, not covered):
//    - conv1x1_gemv_g8_wquant_sg W2/3 multi-token: only selected when the device
//      lacks simdgroup-matrix; otherwise W2/3 prefill takes the outer-dequant path
//      (see MetalConvolution1x1.mm).
//    - fusion variants (GATE_UP_FUSED / QKV_FUSED / LN_FUSED / GEMV_SG_OC_QUADS_2): only built
//      by fusion leaders inside a full LLM graph, not by an isolated conv op.
//
//  Requires MNN_METAL_ENABLED and the low-memory weight-quant path (MNN_LOW_MEMORY).
//
#if defined(MNN_METAL_ENABLED) && MNN_METAL_ENABLED && defined(MNN_LOW_MEMORY)

#include <cmath>
#include <cstring>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "CommonOpCreator.hpp"
#include "core/IDSTEncoder.hpp"
#include "core/TensorUtils.hpp"
#include "backend/metal/MetalConv1x1Decode.hpp"
#include "backend/metal/MetalConv1x1Prefill.hpp"

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
            {512, 256, 1, 1, 32, "2sg decode + split K (oc%8==0)", false},
            {512, 252, 1, 1, 32, "2sg decode no-splitk (oc%8!=0)", false},
            {2048, 64, 1, 1, 32, "2sg decode + Q4 block32", true},
            {2048, 64, 1, 1, 64, "2sg decode + Q4 block64", true},
            {2048, 64, 1, 1, 128, "2sg decode + Q4 block128", true},
            {2048, 64, 1, 1, 256, "2sg decode + Q4 block256", true},
            {2560, 64, 1, 1, 64, "2sg decode + Q4 block64 (40 blocks, split K small grid lanes 1)", true},
            {3584, 64, 1, 1, 64, "2sg decode + Q4 block64 (56 blocks, split K small grid lanes 2)", true},
            {2560, 512, 1, 1, 64, "2sg decode + Q4 block64 (40 blocks, base M5 split K lanes 4)", true},
            {2560, 768, 1, 1, 64, "2sg decode + Q4 block64 (40 blocks, Pro split K lanes 4)", true},
            {2560, 1296, 1, 1, 64, "2sg decode + Q4 block64 (40 blocks, base M5 split K past the TG budget)", true},
            {6144, 64, 1, 1, 64, "2sg decode + Q4 block64 (96 blocks)", true},
            {6144, 1296, 1, 1, 64, "2sg decode + Q4 block64 (96 blocks, Pro tg4 oc>1280)", true},
            // Quant blocks wider than a cache line on the plain route, where
            // gemvPlainLanesPerBlock overrides the kernel's default lane split.
            {4096, 64, 1, 1, 128, "2sg decode tg4 + Q4 block128 (32 blocks, cache-line lanes)", true},
            {4096, 8, 1, 1, 128, "2sg decode 2quad + Q4 block128 (32 blocks, cache-line lanes)", true},
            {8192, 64, 1, 1, 128, "2sg decode tg4 + Q4 block128 (64 blocks, cache-line lanes)", true},
            {128, 16400, 1, 1, 32, "g16 lm_head (oc>16384)", false},
            {2048, 16400, 1, 1, 64, "g16 lm_head + Q4 block64", true},
            {512, 16400, 1, 1, 128, "g16 lm_head + Q4 block128", true},
            {512, 16400, 1, 1, 256, "g16 lm_head + Q4 block256", true},
            // oc % 16 == 0 and an even block count keep decode on the 2sg route,
            // so only the odd-oc shapes above reach g16.
            {2048, 16512, 1, 1, 64, "2sg decode lm_head-sized oc", false},
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
            {640, 64, 1, 2, 128, "fused Q4 K-split M8 (5 blocks, empty last partition)", true},
            {768, 64, 1, 2, 128, "fused Q4 K-split M8 (6 blocks, empty last partition)", true},
            {1152, 64, 1, 9, 128, "fused Q4 K-split M32 (9 blocks, empty last partition)", true},
            // Area 6..8 on tensor-API devices: W4 takes the K-split M8, W8 the plain M8 twin.
            {1024, 64, 1, 6, 64, "fused M8 area6 (W4 ksplit / W8 plain)", false},
            {1024, 64, 1, 7, 128, "fused M8 odd area7 (W4 ksplit / W8 plain)", false},
            {1024, 64, 1, 8, 128, "fused M8 full tile (W4 ksplit / W8 plain)", false},
            {1024, 60, 1, 8, 128, "fused M8 oc%64 tail (W4 ksplit / W8 plain)", false},
            {256, 128, 8, 8, 32, "outer-dequant pre-pass + fp gemm (all bits; area=64, ic*oc<4M)", false},
            {2048, 2176, 1, 2, 32, "g4mN multi-token in-shader (W4/8)", false},
            {2048, 2176, 1, 2, 64, "g4mN multi-token + Q4 block64", true},
            {2048, 2176, 4, 8, 32, "gemm sg-matrix prefill (W4/8)", false},
            {2048, 2176, 4, 8, 64, "gemm sg-matrix prefill + Q4 block64", true},
            // Quant blocks that are not a multiple of 32 K must avoid the 8-quad tile GEMMs.
            {2048, 64, 1, 1, 16, "2sg decode + block16", false},
            {2048, 2176, 1, 2, 16, "g4mN multi-token + block16", false},
            {2048, 2176, 4, 8, 16, "in-shader prefill + block16 (W4/8)", false},
            {256, 128, 8, 8, 16, "prefill area=64 + block16 (all bits)", false},
            {256, 128, 8, 8, 8, "prefill area=64 + block8 (all bits)", false},
            {1536, 128, 8, 8, 48, "prefill area=64 + block48 (all bits)", false},
            {1536, 128, 8, 16, 48, "prefill area=128 + block48 (all bits)", false},
            // Non-power-of-two lanes per block leave spare lanes that must not repeat a block.
            {1536, 64, 1, 1, 48, "2sg decode oc%16==0 + block48", false},
            {1536, 60, 1, 1, 48, "2sg decode no-splitk + block48 (default lanes 3)", false},
            {1536, 128, 1, 2, 48, "g4mN multi-token + block48 (lanes 12)", false},
            {1584, 16400, 1, 1, 48, "g16 lm_head + block48 (33 blocks, lanes 12)", false},
            // heavyMemory (oc>16384, ic*oc>4M): area 4/5 take g4mN unhalved, area 6 falls to the M8 tile.
            {2048, 16400, 1, 4, 64, "g4m4 lm_head-scale verify width (W4/8)", false},
            {2048, 16400, 1, 5, 64, "g4m5 lm_head-scale predicate edge (W4/8)", false},
            {2048, 16400, 1, 6, 64, "M8 tile lm_head-scale past the edge (W4/8)", false},
            {4096, 64, 1, 1, 128, "2sg decode tg4 + block128 (all bits, W3 cache-line lanes)", false},
        };

        bool allPass = true;
        int ran = 0;
        for (auto& s : shapes) {
            // g4mN / in-shader gemm kernels have no true W2/3 branches: area>1 with
            // ic*oc>4M routes W2/3 to outer-dequant instead, covered by the area=64 shape.
            bool skipW23 = (s.ih * s.iw > 1) && ((size_t)s.ic * s.oc > (size_t)4 * 1024 * 1024);
            // Decode routes are shared by Q4 and Q8, so q4Only decode shapes also run W8.
            const bool w8Decode = s.ih * s.iw == 1;
            for (int nbit : {2, 3, 4, 8}) {
                if (s.q4Only && nbit != 4 && !(nbit == 8 && w8Decode)) {
                    continue;
                }
                if ((nbit == 2 || nbit == 3) && skipW23) {
                    continue;
                }
                for (bool async : {false, true}) {
                    bool ok = testUnit(type, s.ic, s.oc, s.ih, s.iw, s.blockSize, nbit, async, precision, s.kernelNote);
                    ++ran;
                    if (!ok) {
                        allPass = false;
                    }
                }
            }
        }
        if (allPass) {
            MNN_PRINT("MetalConvWQuant all %d cases passed.\n", ran);
        }
        return allPass;
    }
};

MNNTestSuiteRegister(MetalConvWQuantTest, "op/conv_wquant_metal");

class MetalConvFp16MetadataTest : public MNNTestCase {
    static bool checkCase(int ic, int oc, int blockSize, int nbits, bool asymmetric, int aMin, float quantScale,
                          int& metalConvs) {
        const int blocks = ic / blockSize, offset = 1 << (nbits - 1);
        const int codeMin = asymmetric ? -offset : 1 - offset;
        const int codeRange = offset - codeMin;
        std::vector<int8_t> codes(ic * oc);
        std::vector<float> alpha((asymmetric ? 2 : 1) * oc * blocks), bias(oc), weight(ic * oc);
        for (int o = 0; o < oc; ++o) {
            bias[o] = (o % 9 - 4) / 64.0f;
            for (int b = 0; b < blocks; ++b) {
                const int k = o * blocks + b;
                const float scale = (0.0937f + 0.0113f * ((o + 3 * b) % 5)) / offset;
                if (asymmetric) {
                    alpha[2 * k] = -0.0737f + 0.0193f * ((o * 3 + b) % 7) + (aMin == 1 ? offset * scale : 0);
                    alpha[2 * k + 1] = scale;
                } else {
                    alpha[k] = scale;
                }
                for (int c = 0; c < blockSize; ++c) {
                    codes[o * ic + b * blockSize + c] = (o * 7 + b * 3 + c * 11) % codeRange + codeMin;
                }
            }
        }
        auto encoded = IDSTEncoder::encode(nullptr, alpha, blockSize, oc * blocks, asymmetric, codes.data(), aMin,
                                           {nbits, false, 16});
        encoded->quantScale = quantScale;
        MNNTEST_ASSERT(encoded->scaleStorage == ScaleStorageType_FP16 && encoded->alpha.empty());
        MNNTEST_ASSERT(encoded->alphaFp16.size() == alpha.size());
        for (int i = 0; i < alpha.size(); ++i) {
            half_float::half h;
            ::memcpy(&h, &encoded->alphaFp16[i], sizeof(uint16_t));
            alpha[i] = float(h);
        }
        // Both twins share encoded codes; FP32 metadata is an exact widening of the serialized FP16 payload.
        OpT twins[2];
        for (int t = 0; t < 2; ++t) {
            twins[t].type = OpType_Convolution;
            twins[t].main.type = OpParameter_Convolution2D;
            twins[t].main.value = new Convolution2DT;
            auto conv = twins[t].main.AsConvolution2D();
            conv->common.reset(new Convolution2DCommonT);
            conv->common->inputCount = ic;
            conv->common->outputCount = oc;
            conv->common->kernelX = conv->common->kernelY = 1;
            conv->bias = bias;
            conv->quanParameter.reset(new IDSTQuanT(*encoded));
            if (t == 1) {
                conv->quanParameter->scaleStorage = ScaleStorageType_FP32;
                conv->quanParameter->alpha = alpha;
                conv->quanParameter->alphaFp16.clear();
            }
        }
        // Apply legacy folding and quantScale in FP32, without rounding corrected coefficients back to half.
        if (asymmetric && aMin <= 0) {
            for (int k = 0; k < oc * blocks; ++k) {
                alpha[2 * k] -= (aMin == 0 ? -128 : aMin) * alpha[2 * k + 1];
            }
        }
        for (auto& v : alpha) {
            v *= quantScale;
        }
        for (int i = 0; i < weight.size(); ++i) {
            const int k = i / blockSize;
            weight[i] = asymmetric ? codes[i] * alpha[2 * k + 1] + alpha[2 * k] : codes[i] * alpha[k];
        }
        for (int area : {1, 64}) {
            std::vector<float> input(ic * area), ref;
            for (int i = 0; i < input.size(); ++i) {
                input[i] = ((i * 13) % 67 - 33) / 64.0f;
            }
            referenceConv1x1(input, weight, bias, ref, ic, oc, area, false, false);
            auto x = _Input({1, ic, 1, area}, NCHW, halide_type_of<float>());
            auto inputPtr = x->writeMap<float>();
            MNNTEST_ASSERT(inputPtr != nullptr);
            ::memcpy(inputPtr, input.data(), input.size() * sizeof(float));
            x->unMap();
            x = _Convert(x, NC4HW4);
            const int before = metalConvs;
            auto yHalf = _Convert(Variable::create(Expr::create(&twins[0], {x})), NCHW);
            auto yFloat = _Convert(Variable::create(Expr::create(&twins[1], {x})), NCHW);
            auto halfPtr = yHalf->readMap<float>();
            auto floatPtr = yFloat->readMap<float>();
            MNNTEST_ASSERT(halfPtr && floatPtr && metalConvs == before + 2);
            MNNTEST_ASSERT(yHalf->getInfo()->size == ref.size() && yFloat->getInfo()->size == ref.size());
            for (int i = 0; i < ref.size(); ++i) {
                if (!std::isfinite(halfPtr[i]) || !std::isfinite(floatPtr[i]) || halfPtr[i] != floatPtr[i]) {
                    MNN_ERROR("Metal FP16 metadata twin mismatch: w%d aMin=%d scale=%g area=%d index=%d\n",
                              nbits, aMin, quantScale, area, i);
                    return false;
                }
            }
            if (!checkVectorByRelativeError(halfPtr, ref.data(), (int)ref.size(), 0.005f)) {
                MNN_ERROR("Metal FP16 metadata reference mismatch: ic=%d oc=%d w%d asym=%d aMin=%d scale=%g area=%d\n",
                          ic, oc, nbits, asymmetric, aMin, quantScale, area);
                return false;
            }
        }
        return true;
    }

public:
    bool run(int precision) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_METAL) {
            MNN_PRINT("Metal FP16 metadata: skipped (requires backend=1).\n");
            return true;
        }
        int metalConvs = 0;
        BackendConfig config;
        config.precision = (BackendConfig::PrecisionMode)precision;
        config.memory = BackendConfig::Memory_Low;
        auto exe = Executor::newExecutor(MNN_FORWARD_METAL, config, 1);
        MNNTEST_ASSERT(exe != nullptr);
        ExecutorScope scope(exe);
        exe->setCallBack([](const std::vector<Tensor*>&, const OperatorInfo*) { return true; },
                         [&metalConvs](const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
            if (info && info->type() == "Convolution" && !outputs.empty()) {
                auto backend = TensorUtils::getDescribeOrigin(outputs[0])->getBackend();
                if (backend && backend->type() == MNN_FORWARD_METAL) {
                    ++metalConvs;
                }
            }
            return true;
        });
        for (int nbits : {2, 3, 4, 8}) {
            MNNTEST_ASSERT(checkCase(64, 17, 32, nbits, false, 1, 1.0f, metalConvs));
            MNNTEST_ASSERT(checkCase(64, 17, 32, nbits, true, 1, 1.0f, metalConvs));
            MNNTEST_ASSERT(checkCase(128, 40, 64, nbits, true, -(1 << (nbits - 1)), 1.0f, metalConvs));
        }
        MNNTEST_ASSERT(checkCase(128, 40, 64, 8, true, 0, 1.0f, metalConvs));
        MNNTEST_ASSERT(checkCase(64, 17, 32, 4, false, 1, 1.7f, metalConvs));
        MNNTEST_ASSERT(checkCase(64, 17, 32, 4, true, 1, 1.7f, metalConvs));
        MNNTEST_ASSERT(checkCase(128, 40, 64, 4, true, -8, 1.7f, metalConvs));
        return true;
    }
};
MNNTestSuiteRegister(MetalConvFp16MetadataTest, "op/conv_wquant_metal/fp16_metadata");

class MetalConvLmHeadTest : public MNNTestCase {
public:
    bool run(int precision) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_METAL) {
            MNN_PRINT("Metal lm_head: skipped (requires backend=1).\n");
            return true;
        }
        for (int ic : {1024, 2048}) {
            if (!runShape(ic, precision)) {
                return false;
            }
        }
        return true;
    }

private:
    bool runShape(int ic, int precision) {
        constexpr int oc = 248320, blockSize = 64;
        const int blocks = ic / blockSize;
        std::vector<float> inputs[2], refs[2];
        for (int t = 0; t < 2; ++t) {
            inputs[t].resize(ic);
            refs[t].resize(oc);
            for (int c = 0; c < ic; ++c) {
                inputs[t][c] = ((c * (13 + 4 * t) + 11 * t) % 67 - 33) / 64.0f;
            }
        }
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_Convolution;
        op->main.type = OpParameter_Convolution2D;
        op->main.value = new Convolution2DT;
        auto conv = op->main.AsConvolution2D();
        conv->common.reset(new Convolution2DCommonT);
        conv->common->inputCount = ic;
        conv->common->outputCount = oc;
        conv->common->kernelX = conv->common->kernelY = 1;
        conv->bias.resize(oc);
        {
            std::vector<int8_t> codes((size_t)ic * oc);
            std::vector<float> alpha((size_t)2 * oc * blocks);
            for (int o = 0; o < oc; ++o) {
                const float bias = (o % 9 - 4) / 64.0f;
                conv->bias[o] = bias;
                double sums[2] = {bias, bias};
                for (int b = 0; b < blocks; ++b) {
                    const size_t k = (size_t)o * blocks + b;
                    const float minimum = ((o + b) % 7 - 8) / 32.0f;
                    const float scale = (1 + (o + 3 * b) % 5) / 256.0f;
                    alpha[2 * k] = minimum;
                    alpha[2 * k + 1] = scale;
                    for (int c = 0; c < blockSize; ++c) {
                        const int channel = b * blockSize + c;
                        const size_t index = (size_t)o * ic + channel;
                        uint32_t hash = (uint32_t)index * 1664525u + 1013904223u;
                        hash ^= hash >> 16;
                        const int code = (int)(hash & 15u) - 8;
                        codes[index] = (int8_t)code;
                        const float weight = (code + 8) * scale + minimum;
                        for (int t = 0; t < 2; ++t) {
                            sums[t] += (double)weight * inputs[t][channel];
                        }
                    }
                }
                for (int t = 0; t < 2; ++t) {
                    refs[t][o] = (float)sums[t];
                }
            }
            conv->quanParameter = IDSTEncoder::encode(
                nullptr, alpha, blockSize, oc * blocks, true, codes.data(), -8, {4, false});
        }
        BackendConfig config;
        config.precision = (BackendConfig::PrecisionMode)precision;
        config.memory = BackendConfig::Memory_Low;
        auto exe = Executor::newExecutor(MNN_FORWARD_METAL, config, 1);
        MNNTEST_ASSERT(exe != nullptr);
        ExecutorScope scope(exe);
        int metalConvs = 0;
        exe->setCallBack([](const std::vector<Tensor*>&, const OperatorInfo*) { return true; },
                        [&metalConvs](const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
            if (info && info->type() == "Convolution" && !outputs.empty()) {
                auto backend = TensorUtils::getDescribeOrigin(outputs[0])->getBackend();
                if (backend && backend->type() == MNN_FORWARD_METAL) {
                    ++metalConvs;
                }
            }
            return true;
        });
        auto x = _Input({1, ic, 1, 1}, NCHW, halide_type_of<float>());
        auto packed = _Convert(x, NC4HW4);
        auto y = _Convert(Variable::create(Expr::create(std::move(op), {packed})), NCHW);
        for (int t = 0; t < 2; ++t) {
            auto input = x->writeMap<float>();
            MNNTEST_ASSERT(input != nullptr);
            ::memcpy(input, inputs[t].data(), ic * sizeof(float));
            x->unMap();
            const int before = metalConvs;
            auto output = y->readMap<float>();
            MNNTEST_ASSERT(output && y->getInfo()->size == oc && metalConvs == before + 1);
            const float tolerance = precision == BackendConfig::Precision_High ? 0.00001f : 0.005f;
            for (int o = 0; o < oc; ++o) {
                if (!std::isfinite(output[o]) ||
                    std::fabs(output[o] - refs[t][o]) > tolerance * (1.0f + std::fabs(refs[t][o]))) {
                    MNN_ERROR("Metal lm_head reference failed: input=%d oc=%d got=%g ref=%g\n",
                              t, o, output[o], refs[t][o]);
                    return false;
                }
            }
            y->unMap();
        }
        MNN_PRINT("Metal lm_head oracle passed: ic=%d oc=%d block=%d inputs=2 precision=%d\n",
                  ic, oc, blockSize, precision);
        return true;
    }
};
MNNTestSuiteRegister(MetalConvLmHeadTest, "op/conv_wquant_metal/lm_head");

// The Q4 four-way K-split re-creates mKsplitPartial on every size change but keeps the same pipeline,
// so a surviving recording would hold a freed Tensor* and a stale grid.
class MetalConvReplaySwitchTest : public MNNTestCase {
public:
    virtual ~MetalConvReplaySwitchTest() = default;

    bool runSequence(int ic, int oc, int blockSize, const std::vector<int>& areas, const char* tag, int precision,
                     int bits = 4) {
        std::vector<float> weight((size_t)oc * ic);
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = ((float)((i * 1103515245u + 12345u) % 65536) / 65536.0f) - 0.5f;
        }
        std::vector<float> bias(oc);
        for (int o = 0; o < oc; ++o) {
            bias[o] = ((float)((o * 2654435761u) % 65536) / 65536.0f) - 0.5f;
        }
        std::vector<float> alpha;
        quantizeDequantize(weight, alpha, ic, oc, blockSize, bits, true);

        // StaticModule keeps the Execution across resizes; a lazy Express graph rebuilds it and hides the hazard.
        auto x0 = _Input({1, ic, 1, areas[0]}, NCHW, halide_type_of<float>());
        auto y0 = _HybridConv(weight, bias, alpha, x0, {ic, oc}, {1, 1}, CAFFE, {1, 1}, {1, 1}, 1, {0, 0}, false, false,
                              bits, true);
        y0 = _Convert(y0, NCHW);
        auto buffer = Variable::save({y0});
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_METAL;
        config.numThread = 1;
        BackendConfig bnConfig;
        bnConfig.precision = (BackendConfig::PrecisionMode)precision;
        bnConfig.memory = BackendConfig::Memory_Low;
        config.backendConfig = &bnConfig;
        std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
        std::shared_ptr<Module> module(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
        if (nullptr == module) {
            MNN_ERROR("ConvReplaySwitch: module load failed (%s)\n", tag);
            return false;
        }

        for (int area : areas) {
            std::vector<float> input((size_t)ic * area);
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = ((float)((i * 40503u) % 65536) / 65536.0f) - 0.5f;
            }
            std::vector<float> ref;
            referenceConv1x1(input, weight, bias, ref, ic, oc, area, false, false);
            auto x = _Input({1, ic, 1, area}, NCHW, halide_type_of<float>());
            ::memcpy(x->writeMap<float>(), input.data(), input.size() * sizeof(float));
            x->unMap();
            auto runOnce = [&](const std::vector<float>& expect, int rep) {
                VARP out = module->onForward({x})[0];
                auto ptr = out->readMap<float>();
                if (nullptr == ptr) {
                    MNN_ERROR("ConvReplaySwitch: readMap null area=%d\n", area);
                    return false;
                }
                bool ok = checkVectorByRelativeError<float>(ptr, expect.data(), (int)expect.size(), 0.005f);
                out->unMap();
                if (!ok) {
                    MNN_ERROR("ConvReplaySwitch FAILED: %s area=%d rep=%d ic=%d oc=%d block=%d\n", tag, area, rep, ic,
                              oc, blockSize);
                }
                return ok;
            };
            for (int rep = 0; rep <= 3; ++rep) {
                if (!runOnce(ref, rep)) {
                    return false;
                }
            }
            // Same shape and buffer, new content: a stale replay must not serve the recorded result.
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = ((float)((i * 22695477u + 1u) % 65536) / 65536.0f) - 0.5f;
            }
            std::vector<float> ref2;
            referenceConv1x1(input, weight, bias, ref2, ic, oc, area, false, false);
            ::memcpy(x->writeMap<float>(), input.data(), input.size() * sizeof(float));
            x->unMap();
            for (int rep = 4; rep <= 5; ++rep) {
                if (!runOnce(ref2, rep)) {
                    return false;
                }
            }
        }
        MNN_PRINT("ConvReplaySwitch passed: %s (ic=%d oc=%d block=%d, %d areas x6)\n", tag,
                  ic, oc, blockSize, (int)areas.size());
        return true;
    }

    virtual bool run(int precision) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_METAL) {
            MNN_PRINT("ConvReplaySwitch: skipped (Metal-only test, run with backend=1).\n");
            return true;
        }
        MNN_PRINT("\n## ConvReplaySwitch (precision=%d)\n", precision);
        // blockCount 9 >= 4 and UP_DIV(oc,64) <= 48 keep the four-way K-split on.
        return runSequence(1152, 64, 128, {16, 24, 32, 16}, "M32 ksplit partial growth", precision) &&
               runSequence(1152, 64, 128, {2, 4, 6, 7, 8, 2}, "M8 ksplit partial growth", precision) &&
               runSequence(1152, 64, 128, {1, 16, 24, 1, 32, 16}, "decode/prefill route switch", precision) &&
               runSequence(1024, 64, 128, {1, 4, 64, 512, 1}, "W4 prefill stages", precision) &&
               runSequence(1024, 64, 128, {1, 4, 64, 512, 1}, "W8 prefill stages", precision, 8) &&
               runSequence(1024, 64, 16, {1, 4, 64, 512, 1}, "SG block-alignment fallback", precision);
    }
};
MNNTestSuiteRegister(MetalConvReplaySwitchTest, "op/conv_replay_switch_metal");

// Area 2..5 takes the in-shader g4mN GEMV ladder by default; larger areas cover the tile chain instead.
// Cases span the piece=1/2 halving boundary, odd areas and lane counts, and heavyMemory which blocks halving.
class MetalConvPrefillGemvTest : public MNNTestCase {
public:
    virtual ~MetalConvPrefillGemvTest() = default;

    virtual bool run(int precision) override {
        auto status = MNNTestSuite::get()->pStaus;
        if (status.forwardType != MNN_FORWARD_METAL) {
            MNN_PRINT("ConvPrefillGemv: skipped (Metal-only test, run with backend=1).\n");
            return true;
        }
        BackendConfig bnConfig;
        bnConfig.precision = (BackendConfig::PrecisionMode)precision;
        bnConfig.memory    = BackendConfig::Memory_Low;
        auto exe = Executor::newExecutor(MNN_FORWARD_METAL, bnConfig, 1);
        ExecutorScope scope(exe);
        MNN_PRINT("\n## ConvPrefillGemv (precision=%d)\n", precision);

        struct Case { int ic, oc, area, blockSize, nbit; bool async; };
        const Case cases[] = {
            {1024, 64,    2,  64,  4, true},
            {1024, 64,    3,  64,  4, true},
            {1024, 64,    5,  64,  4, true},
            {1024, 64,    7,  64,  4, true},
            {1024, 64,    8,  64,  4, true},
            {1024, 64,    12, 64,  4, true},
            {1024, 64,    15, 64,  4, true},
            {1024, 64,    16, 64,  4, true},
            {1152, 128,   9,  128, 4, true},
            {1536, 60,    6,  48,  4, false},
            {2048, 2176,  4,  64,  8, true},
            {2048, 2176,  16, 64,  4, true},
            {1024, 64,    8,  64,  8, true},
            {1024, 64,    8,  64,  2, true},
            {1024, 64,    8,  64,  3, true},
            {1024, 16512, 8,  64,  4, true},
            // At lm_head scale the g4mN branch preempts the g16 kernel that area 1 uses.
            {1024, 16512, 4,  64,  4, true},
            {1024, 16512, 5,  64,  4, true},
            // Real Qwen3-0.6B verify-width projections (q_proj / down).
            {1024, 2048,  4,  64,  4, true},
            {3072, 1024,  4,  64,  4, true},
            // oc%4!=0 tail threadgroup, non-power-of-two lanesPerBlock (12).
            {1536, 62,    4,  48,  4, false},
            // Odd blockCount 9 at piece=1: leftover-lane handling without the halving.
            {1152, 128,   4,  128, 4, true},
            // W8B64 is a shipped config: gate_up and lm_head scale at the verify width.
            {1024, 3072,  4,  64,  8, true},
            {1024, 16512, 4,  64,  8, true},
        };
        MetalConvWQuantTest unit;
        int n = 0;
        for (const auto& c : cases) {
            if (!unit.testUnit(MNN_FORWARD_METAL, c.ic, c.oc, 1, c.area, c.blockSize, c.nbit, c.async, precision,
                               "prefill-gemv")) {
                MNN_ERROR("ConvPrefillGemv FAILED: ic=%d oc=%d area=%d block=%d w%d async=%d\n", c.ic, c.oc, c.area,
                          c.blockSize, c.nbit, (int)c.async);
                return false;
            }
            ++n;
        }
        MNN_PRINT("ConvPrefillGemv passed: %d prefill shapes x3 activations\n", n);
        return true;
    }
};
MNNTestSuiteRegister(MetalConvPrefillGemvTest, "op/conv_prefill_gemv_metal");

// MetalFusedProj::onEncode calls the children directly, so a child's scratch re-creation is invisible to
// the owner's generation counter and to metalReplayValidate; the owner must consult the children first.
class MetalFusedProjReplaySwitchTest : public MNNTestCase {
public:
    virtual ~MetalFusedProjReplaySwitchTest() = default;

    // FusedLinear keeps tokens in dim 0 with NC4HW4 storage: plain [seq][channel] packs as [c/4][seq][4].
    static std::vector<float> packC4(const std::vector<float>& plain, int seq, int channel) {
        std::vector<float> out(((channel + 3) / 4) * seq * 4, 0.0f);
        for (int t = 0; t < seq; ++t) {
            for (int c = 0; c < channel; ++c) {
                out[(c / 4) * seq * 4 + t * 4 + (c % 4)] = plain[t * channel + c];
            }
        }
        return out;
    }

    // `weight` stays dequantized for the fp32 reference.
    static std::unique_ptr<Convolution2DT> member(int ic, int oc, int blockSize, int seed,
                                                  std::vector<float>& weight, std::vector<float>& bias,
                                                  int nbit = 4) {
        weight.resize((size_t)oc * ic);
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = ((float)(((i + (size_t)seed) * 1103515245u + 12345u) % 65536) / 65536.0f) - 0.5f;
        }
        bias.resize(oc);
        for (int o = 0; o < oc; ++o) {
            bias[o] = ((float)(((o + seed) * 2654435761u) % 65536) / 65536.0f) - 0.5f;
        }
        std::vector<float> alpha;
        quantizeDequantize(weight, alpha, ic, oc, blockSize, nbit, true);
        std::unique_ptr<Convolution2DT> conv(new Convolution2DT);
        conv->common.reset(new Convolution2DCommonT);
        conv->common->kernelX     = 1;
        conv->common->kernelY     = 1;
        conv->common->inputCount  = ic;
        conv->common->outputCount = oc;
        conv->bias                = bias;
        const int clampMin        = -(1 << (nbit - 1));
        conv->quanParameter = IDSTEncoder::encode(weight.data(), alpha, blockSize, oc * (ic / blockSize), true,
                                                  nullptr, clampMin, {nbit, false});
        return conv;
    }

    // x: [seq][ic], w: [oc][ic] -> out: [seq][oc]
    static void convRef(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& b,
                        std::vector<float>& out, int seq, int ic, int oc) {
        out.assign((size_t)seq * oc, 0.0f);
        for (int t = 0; t < seq; ++t) {
            for (int o = 0; o < oc; ++o) {
                float sum = b[o];
                for (int i = 0; i < ic; ++i) {
                    sum += w[(size_t)o * ic + i] * x[(size_t)t * ic + i];
                }
                out[(size_t)t * oc + o] = sum;
            }
        }
    }

    bool runGroup(bool gateUp, const std::vector<int>& areas, const char* tag, int precision, int nbit = 4,
                  int qkvMembers = 3) {
        const int ic = 1152, oc = 64, blockSize = 128;
        const int numMembers = gateUp ? 2 : qkvMembers;
        const int numOuts    = gateUp ? 1 : numMembers;
        std::vector<std::vector<float>> weights(numMembers), biases(numMembers);
        std::unique_ptr<OpT> op(new OpT);
        op->type                   = OpType_FusedLinear;
        op->main.type              = OpParameter_FusedLinearParam;
        op->main.value             = new FusedLinearParamT;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto param                 = op->main.AsFusedLinearParam();
        param->act_silu_mul        = gateUp;
        param->has_ln              = false;
        for (int i = 0; i < numMembers; ++i) {
            param->convs.push_back(member(ic, oc, blockSize, 17 * i + 3, weights[i], biases[i], nbit));
        }

        auto x0   = _Input({areas[0], ic, 1, 1}, NC4HW4);
        auto expr = Expr::create(std::move(op), {x0}, numOuts);
        std::vector<VARP> ys;
        for (int i = 0; i < numOuts; ++i) {
            ys.push_back(Variable::create(expr, i));
        }
        auto buffer = Variable::save(ys);
        MNN::ScheduleConfig config;
        config.type      = MNN_FORWARD_METAL;
        config.numThread = 1;
        BackendConfig bnConfig;
        bnConfig.precision = (BackendConfig::PrecisionMode)precision;
        bnConfig.memory    = BackendConfig::Memory_Low;
        config.backendConfig = &bnConfig;
        std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
        std::shared_ptr<Module> module(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
        if (nullptr == module) {
            MNN_ERROR("FusedProjReplaySwitch: module load failed (%s)\n", tag);
            return false;
        }

        for (int area : areas) {
            std::vector<float> input((size_t)ic * area);
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = ((float)((i * 40503u) % 65536) / 65536.0f) - 0.5f;
            }
            std::vector<std::vector<float>> refs(numMembers);
            for (int i = 0; i < numMembers; ++i) {
                convRef(input, weights[i], biases[i], refs[i], area, ic, oc);
            }
            std::vector<std::vector<float>> expected(numOuts);
            if (gateUp) {
                // MUL_SILU convention: in0 = up (member 1), in1 = gate (member 0).
                expected[0].resize((size_t)area * oc);
                for (size_t k = 0; k < expected[0].size(); ++k) {
                    const float g  = refs[0][k];
                    expected[0][k] = refs[1][k] * (g / (1.0f + std::exp(-g)));
                }
            } else {
                expected = refs;
            }
            std::vector<std::vector<float>> packed(numOuts);
            for (int i = 0; i < numOuts; ++i) {
                packed[i] = packC4(expected[i], area, oc);
            }

            auto x = _Input({area, ic, 1, 1}, NC4HW4);
            auto xC4 = packC4(input, area, ic);
            ::memcpy(x->writeMap<float>(), xC4.data(), xC4.size() * sizeof(float));
            x->unMap();
            // Two stable fingerprints arm the recording on the third forward; the fourth replays it.
            for (int rep = 0; rep < 4; ++rep) {
                auto outs = module->onForward({x});
                if ((int)outs.size() != numOuts) {
                    MNN_ERROR("FusedProjReplaySwitch: %s area=%d got %zu outputs\n", tag, area, outs.size());
                    return false;
                }
                for (int i = 0; i < numOuts; ++i) {
                    auto ptr = outs[i]->readMap<float>();
                    if (nullptr == ptr) {
                        MNN_ERROR("FusedProjReplaySwitch: readMap null %s area=%d out=%d\n", tag, area, i);
                        return false;
                    }
                    const float tol = (precision == 2) ? 0.01f : 0.002f;
                    if (!checkVector<float>(ptr, packed[i].data(), (int)packed[i].size(), tol)) {
                        MNN_ERROR("FusedProjReplaySwitch FAILED: %s area=%d rep=%d out=%d\n", tag, area, rep, i);
                        return false;
                    }
                    outs[i]->unMap();
                }
            }
        }
        MNN_PRINT("FusedProjReplaySwitch passed: %s (%d members, %d areas x4)\n", tag, numMembers, (int)areas.size());
        return true;
    }

    virtual bool run(int precision) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_METAL) {
            MNN_PRINT("FusedProjReplaySwitch: skipped (Metal-only test, run with backend=1).\n");
            return true;
        }
        MNN_PRINT("\n## FusedProjReplaySwitch (precision=%d)\n", precision);
        // Area >= 128 reaches the gate/up dual fold, where the gate child never encodes.
        // Area 4 loses decode-fusion eligibility to the g4mN ladder, so the owner must switch structure there.
        return runGroup(false, {16, 24, 32, 128, 16}, "QKV no-LN (owner generation never moves)", precision) &&
               runGroup(false, {1, 2, 3, 4, 5, 8, 16, 128, 1, 4, 24, 128, 16}, "QKV decode/prefill switch", precision) &&
               runGroup(false, {1, 2, 3, 4, 5, 8, 16, 32, 128, 1, 4}, "QKV 4-proj short/wide/decode switch", precision, 4, 4) &&
               runGroup(true, {16, 24, 32, 128, 192, 16}, "GateUp silu-mul/dual fold", precision) &&
               runGroup(true, {1, 2, 3, 4, 5, 8, 16, 128, 1, 192, 128, 4, 16}, "GateUp decode/prefill switch", precision) &&
               runGroup(false, {48, 128}, "QKV wide prefill W8", precision, 8) &&
               runGroup(true, {128, 192}, "GateUp dual prefill W8", precision, 8) &&
               runGroup(false, {48, 128}, "QKV wide prefill W2", precision, 2) &&
               runGroup(true, {128, 192}, "GateUp dual prefill W2", precision, 2) &&
               runGroup(false, {48, 128}, "QKV wide prefill W3", precision, 3) &&
               runGroup(true, {128, 192}, "GateUp dual prefill W3", precision, 3);
    }
};
MNNTestSuiteRegister(MetalFusedProjReplaySwitchTest, "op/fused_proj_replay_switch_metal");

// Gates the in-kernel 32x64 wquant split-k tile lowered to area 64-127 small weight
// (non-tensor regime). Run with the local probe forcing non-tensor + wquant small.
class MetalConvWqSmallTileTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto status = MNNTestSuite::get()->pStaus;
        if (status.forwardType != MNN_FORWARD_METAL) {
            MNN_PRINT("ConvWqSmallTile: skipped (Metal-only, backend=1).\n");
            return true;
        }
        BackendConfig bnConfig;
        bnConfig.precision = (BackendConfig::PrecisionMode)precision;
        bnConfig.memory = BackendConfig::Memory_Low;
        auto exe = Executor::newExecutor(MNN_FORWARD_METAL, bnConfig, 1);
        ExecutorScope scope(exe);
        MNN_PRINT("\n## ConvWqSmallTile (precision=%d)\n", precision);
        struct Case { int ic, oc, area, blockSize; };
        const Case cases[] = {
            {1024, 1024, 64, 64},  {1024, 1024, 96, 64},  {1024, 1024, 127, 64}, {1024, 1024, 128, 64},
            {1536, 1536, 96, 64},  {2048, 2048, 64, 64},  {2048, 2048, 96, 64},  {2048, 2048, 127, 64},
            {1024, 2048, 96, 64},  {2048, 1024, 96, 64},
        };
        MetalConvWQuantTest unit;
        int n = 0;
        for (auto& c : cases) {
            for (int nbit : {4, 8}) {
                for (bool async : {false, true}) {
                    if (!unit.testUnit(MNN_FORWARD_METAL, c.ic, c.oc, 1, c.area, c.blockSize, nbit, async, precision,
                                       "wq-small-tile")) {
                        MNN_ERROR("ConvWqSmallTile FAILED: ic=%d oc=%d area=%d block=%d w%d async=%d\n", c.ic, c.oc,
                                  c.area, c.blockSize, nbit, (int)async);
                        return false;
                    }
                    ++n;
                }
            }
        }
        MNN_PRINT("ConvWqSmallTile passed: %d cases\n", n);
        return true;
    }
};
MNNTestSuiteRegister(MetalConvWqSmallTileTest, "op/conv_wq_small_tile_metal");

class MetalConvPrefillEdgeTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        // Pure routing & descriptor regressions (platform-independent).
        {
            PrefillCaps caps;
            caps.allowTensorApi = false;
            caps.hasSgMatrix = true;
            caps.hasSgReduce = true;
            caps.lowMemoryKernelsAvailable = true;
            caps.gpuGen = 15;
            caps.maxThreadgroupMemoryBytes = 32768;

            // area * oc overflows signed 32-bit int; must still select SgInKernel32x64.
            QuantPrefillShape huge;
            huge.area = 16384;
            huge.ic = 2048;
            huge.oc = 151936;
            huge.blockCount = 16;
            huge.dequantBits = 4;
            huge.fp16Storage = true;
            auto hugePlan = selectSimdInKernelPrefill(caps, huge);
            MNNTEST_ASSERT(hugePlan.kernel == PrefillKernel::SgInKernel32x64);

            // SgInKernel32x64 needs 768 * 16 = 12288 B of threadgroup memory.
            PrefillCaps smallTg = caps;
            smallTg.maxThreadgroupMemoryBytes = 8192;
            auto smallTgPlan = selectSimdInKernelPrefill(smallTg, huge);
            MNNTEST_ASSERT(smallTgPlan.kernel != PrefillKernel::SgInKernel32x64);

            // Outer-dequant grids must cover padded C4 rows when oc % 4 != 0.
            QuantPrefillShape unalignedOc;
            unalignedOc.area = 64;
            unalignedOc.ic = 64;
            unalignedOc.oc = 65;
            unalignedOc.blockCount = 1;
            unalignedOc.dequantBits = 4;
            unalignedOc.fp16Storage = true;
            QuantPrefillPlan rowPlan;
            rowPlan.kernel = PrefillKernel::TensorRowDeviceM64;
            auto rowDesc = describeQuantPrefill(rowPlan, unalignedOc);
            MNNTEST_ASSERT(rowDesc.dequant.grid.x == 68);
            QuantPrefillPlan packedPlan;
            packedPlan.kernel = PrefillKernel::SgPreDequantM32;
            auto packedDesc = describeQuantPrefill(packedPlan, unalignedOc);
            MNNTEST_ASSERT(packedDesc.dequant.grid.x == 68);

            // Large pack grid must not overflow 32-bit int on ic * area.
            QuantPrefillShape hugePack = unalignedOc;
            hugePack.area = 16384;
            hugePack.ic = 131072;
            auto hugePackDesc = describeQuantPrefill(rowPlan, hugePack);
            const size_t expectedPackX = ((size_t)(hugePack.ic / 4) * hugePack.area + 127) / 128;
            MNNTEST_ASSERT(hugePackDesc.pack.grid.x == expectedPackX);

            // Tensor M/N tile and K-split routing invariants across W2/W3/W4/W8.
            PrefillCaps tensorCaps = caps;
            tensorCaps.allowTensorApi = true;
            tensorCaps.gpuGen = 16;
            for (int bits : {2, 3, 4, 8}) {
                QuantPrefillShape s;
                s.ic = 2048;
                s.oc = 1024;
                s.blockCount = 32;
                s.dequantBits = bits;
                s.fp16Storage = true;

                s.area = 4;
                MNNTEST_ASSERT(selectQuantPrefill(tensorCaps, s).kernel == PrefillKernel::SgReduceG4mN);
                s.area = 8;
                auto p8 = selectQuantPrefill(tensorCaps, s);
                MNNTEST_ASSERT(p8.kernel == PrefillKernel::TensorInKernel8x64 && p8.kPartitions == 8);
                MNNTEST_ASSERT(p8.loopK64);
                s.area = 12;
                MNNTEST_ASSERT(selectQuantPrefill(tensorCaps, s).kernel == PrefillKernel::TensorInKernel16x64);
                s.area = 24;
                MNNTEST_ASSERT(selectQuantPrefill(tensorCaps, s).kernel == PrefillKernel::TensorInKernel32x64);
                s.area = 48;
                auto p48 = selectQuantPrefill(tensorCaps, s);
                MNNTEST_ASSERT(p48.kernel == PrefillKernel::TensorInKernel64x64 && p48.kPartitions == 1);

                // Small oc <= 32 selects narrow N=32 tile + K-split=8 when blocks allow.
                s.area = 64;
                s.oc = 32;
                auto pN32 = selectQuantPrefill(tensorCaps, s);
                MNNTEST_ASSERT(pN32.kernel == PrefillKernel::TensorInKernel64x64 && pN32.tileN == 32 &&
                               pN32.kPartitions == 8);
                auto dN32 = describeQuantPrefill(pN32, s);
                MNNTEST_ASSERT(dN32.main.grid.x == 1 && dN32.main.grid.y == 1 && dN32.main.grid.z == 8);

                // Small gridXY with blockCount == 1 cannot K-split, so it falls back to N=32.
                s.oc = 256;
                s.blockCount = 1;
                auto pNoKs = selectQuantPrefill(tensorCaps, s);
                MNNTEST_ASSERT(pNoKs.tileN == 32 && pNoKs.kPartitions == 1);
                auto dNoKs = describeQuantPrefill(pNoKs, s);
                MNNTEST_ASSERT(dNoKs.main.grid.y == 8 && dNoKs.main.grid.z == 1);

                // TG-starved multi-block shape triggers K-split=8 (gridXY=4) and K-split=2 (blockCount=2).
                s.blockCount = 32;
                MNNTEST_ASSERT(selectQuantPrefill(tensorCaps, s).kPartitions == 8);
                s.blockCount = 2;
                MNNTEST_ASSERT(selectQuantPrefill(tensorCaps, s).kPartitions == 2);
                // GateUpDual / SiluMul fusion is enabled across all W2/W3/W4/W8 on M64.
                s.area = 128;
                s.oc = 1024;
                s.blockCount = 32;
                s.dualOffer = true;
                s.gateUpDualMinArea = 128;
                MNNTEST_ASSERT(selectQuantPrefill(tensorCaps, s).fusion == PrefillFusion::GateUpDual);
                s.dualOffer = false;

                // FP32 gets FQ_PREFETCH on K=32 (W2/W4/W8) and K=64 on M=8 N=32.
                s.fp16Storage = false;
                s.area = 64;
                auto pFp32 = selectQuantPrefill(tensorCaps, s);
                auto dFp32 = describeQuantPrefill(pFp32, s);
                const bool hasPrefetch = std::find(dFp32.main.keys.begin(), dFp32.main.keys.end(), "FQ_PREFETCH") !=
                                         dFp32.main.keys.end();
                MNNTEST_ASSERT(hasPrefetch == (bits != 3));
                s.area = 8;
                s.oc = 32;
                auto pFp32M8N32 = selectQuantPrefill(tensorCaps, s);
                MNNTEST_ASSERT(pFp32M8N32.loopK64 && pFp32M8N32.tileN == 32);
            }

            // M=8 W4B64 split-K threshold matrix (mirrors M5 Pro / M4 Pro verify shapes):
            // gridXY <= 8 -> KS=8; gridXY <= 32 & ic >= 2048 -> KS=8 (ic < 2048 -> KS=4);
            // gridXY <= 48 -> KS=4; gridXY > 48 (N >= 4096) -> 1-pass KS=1 + TK64.
            {
                PrefillCaps gen17Caps = tensorCaps;
                gen17Caps.gpuGen = 17;
                struct M8PlanCheck { int oc, ic, expKs, expTileN; };
                const M8PlanCheck checks[] = {
                    {16,    2048, 8, 32},
                    {32,    2048, 8, 32},
                    {64,    2048, 8, 0},
                    {512,   2048, 8, 0},
                    {1024,  1024, 4, 0},
                    {1024,  2048, 8, 0},
                    {1024,  3072, 8, 0},
                    {2048,  1024, 4, 0},
                    {2048,  2048, 8, 0},
                    {3072,  2048, 4, 0},
                    {4096,  2048, 1, 0},
                    {6144,  1024, 1, 0},
                    {6144,  2048, 1, 0},
                    {12288, 4096, 1, 0},
                };
                for (const auto& ck : checks) {
                    QuantPrefillShape sm;
                    sm.area = 8;
                    sm.ic = ck.ic;
                    sm.oc = ck.oc;
                    sm.blockCount = ck.ic / 64;
                    sm.dequantBits = 4;
                    sm.fp16Storage = true;
                    auto pm = selectQuantPrefill(gen17Caps, sm);
                    MNNTEST_ASSERT(pm.kernel == PrefillKernel::TensorInKernel8x64);
                    MNNTEST_ASSERT(pm.loopK64 && !pm.hoistW4Scale);
                    MNNTEST_ASSERT(pm.kPartitions == ck.expKs && pm.tileN == ck.expTileN);
                }
            }

            // Non-Tensor short prefill (area 6..8, oc >= 512) uses SgInKernel8x16SplitK across W2/W3/W4/W8;
            // Tensor API stays on TensorInKernel8x64 and small oc / tight smem fall back cleanly.
            for (int bits : {2, 3, 4, 8}) {
                QuantPrefillShape s8;
                s8.area = 8;
                s8.ic = 2048;
                s8.oc = 1024;
                s8.blockCount = 32;
                s8.dequantBits = bits;
                s8.fp16Storage = true;
                MNNTEST_ASSERT(selectQuantPrefill(tensorCaps, s8).kernel == PrefillKernel::TensorInKernel8x64);
                MNNTEST_ASSERT(selectQuantPrefill(caps, s8).kernel == PrefillKernel::SgInKernel8x16SplitK);
                s8.area = 6;
                MNNTEST_ASSERT(selectQuantPrefill(caps, s8).kernel == PrefillKernel::SgInKernel8x16SplitK);
                s8.area = 5;
                MNNTEST_ASSERT(selectQuantPrefill(caps, s8).kernel != PrefillKernel::SgInKernel8x16SplitK);
                s8.area = 8;
                s8.oc = 256;
                MNNTEST_ASSERT(selectQuantPrefill(caps, s8).kernel != PrefillKernel::SgInKernel8x16SplitK);
                s8.oc = 1024;
                s8.fp16Storage = false;
                PrefillCaps smem16k = caps;
                smem16k.maxThreadgroupMemoryBytes = 16384;
                MNNTEST_ASSERT(selectQuantPrefill(smem16k, s8).kernel != PrefillKernel::SgInKernel8x16SplitK);
                MNNTEST_ASSERT(selectQuantPrefill(caps, s8).kernel == PrefillKernel::SgInKernel8x16SplitK);
            }

            // Decode GEMV: oc % 8 == 0 && oc % 16 != 0 at quadsPerBlock >= 32 should use Oc8Nsg4SplitK2, not Plain.
            DecodeCaps decCaps;
            auto decProfile = conv1x1ProfileFrom(Conv1x1DieClass::Pro, 16);
            decCaps.tier = decProfile.tier;
            decCaps.tgBudget = decProfile.tgBudget;
            decCaps.wqEvenBlockFp16 = true;
            DecodeShape decShape;
            decShape.oc = 40;
            decShape.icDiv4 = 256;
            decShape.blockCount = 8; // quadsPerBlock = 32
            decShape.dequantBits = 4;
            MNNTEST_ASSERT(selectDecodeGemv(decCaps, decShape).variant == DecodeGemvVariant::Oc8Nsg4SplitK2);
        }

        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_METAL) {
            MNN_PRINT("ConvPrefillEdge: skipped GPU cases (Metal-only, backend=1).\n");
            return true;
        }
        BackendConfig bnConfig;
        bnConfig.precision = (BackendConfig::PrecisionMode)precision;
        bnConfig.memory = BackendConfig::Memory_Low;
        auto exe = Executor::newExecutor(MNN_FORWARD_METAL, bnConfig, 1);
        ExecutorScope scope(exe);
        MNN_PRINT("\n## ConvPrefillEdge (precision=%d)\n", precision);

        // Exercises MNN_METAL_SRC_PROTECT weight-tail guards (icDiv4 % 8 != 0, ic < 32,
        // multi-block uneven last block), oc % 4 != 0 padding, M=8/16/32/64 & N=32 tiles,
        // K-split=2/4/8 across W2/W3/W4/W8, and decode oc % 8 == 0 && oc % 16 != 0.
        struct Case { int ic, oc, area, blockSize; };
        const Case cases[] = {
            {12,  17,  7,  12},
            {20,  65,  8,  20},
            {48,  65,  16, 48},
            {48,  17,  64, 48},
            {100, 65,  64, 100},
            {124, 65,  64, 62},
            {256, 32,  12, 32},   // M16 x N32 + KS=8
            {256, 128, 16, 32},   // M16 x N64 + KS=8
            {256, 65,  48, 32},   // M64 x N64 + KS=8 + oc%4!=0
            {128, 256, 48, 64},   // M64 x N64 + KS=2
            {128, 128, 64, 128},  // M64 x N32 (blockCount=1 narrow-N fallback)
            {256, 32,  64, 32},   // M64 x N32 + KS=8
            {256, 512, 8,  64},   // Non-Tensor SgInKernel8x16SplitK (full K tile)
            {100, 517, 7,  100},  // Non-Tensor SgInKernel8x16SplitK (partial M/K/N tails)
            {256, 512, 6,  64},   // Non-Tensor SgInKernel8x16SplitK (M=6 tail)
            {256, 40,  1,  128},  // Decode Oc8Nsg4SplitK2 (oc%8==0, oc%16!=0, quadsPerBlock=32)
            {256, 128, 2,  64},   // SgReduceG4mN (area=2 zero-shift W4 / W8)
            {256, 128, 3,  64},   // SgReduceG4mN (area=3)
            {256, 128, 5,  64},   // SgReduceG4mN (area=5)
            {128, 65,  33, 64},   // Non-aligned M=33 (32x16 non-Tensor & M64 Tensor)
            {128, 65,  47, 64},   // Non-aligned M=47 (32x16 non-Tensor & M64 Tensor)
            {128, 548, 20, 64},   // Non-aligned oc%32!=0 (16x32 non-Tensor)
        };
        MetalConvWQuantTest unit;
        int n = 0;
        for (const auto& c : cases) {
            for (int nbit : {2, 3, 4, 8}) {
                for (bool async : {false, true}) {
                    if (!unit.testUnit(MNN_FORWARD_METAL, c.ic, c.oc, 1, c.area, c.blockSize, nbit, async, precision,
                                       "prefill-edge")) {
                        MNN_ERROR("ConvPrefillEdge FAILED: ic=%d oc=%d area=%d block=%d w%d async=%d\n",
                                  c.ic, c.oc, c.area, c.blockSize, nbit, (int)async);
                        return false;
                    }
                    ++n;
                }
            }
        }
        // Full M=8 W4B64 LLM verify shape matrix (covering N=16..12288, K=1024..4096).
        const Case llmM8Cases[] = {
            {2048, 16,    8, 64},
            {2048, 32,    8, 64},
            {2048, 64,    8, 64},
            {2048, 128,   8, 64},
            {2048, 256,   8, 64},
            {2048, 512,   8, 64},
            {1024, 1024,  8, 64},
            {2048, 1024,  8, 64},
            {3072, 1024,  8, 64},
            {2048, 2048,  8, 64},
            {2048, 3072,  8, 64},
            {2048, 4096,  8, 64},
            {1024, 6144,  8, 64},
            {2048, 6144,  8, 64},
            {4096, 12288, 8, 64},
        };
        for (const auto& c : llmM8Cases) {
            if (!unit.testUnit(MNN_FORWARD_METAL, c.ic, c.oc, 1, c.area, c.blockSize, 4, true, precision,
                               "prefill-m8-llm")) {
                MNN_ERROR("ConvPrefillEdge M8 LLM FAILED: ic=%d oc=%d area=%d block=%d\n",
                          c.ic, c.oc, c.area, c.blockSize);
                return false;
            }
            ++n;
        }
        MNN_PRINT("ConvPrefillEdge passed: %d cases\n", n);
        return true;
    }
};
MNNTestSuiteRegister(MetalConvPrefillEdgeTest, "op/conv_prefill_edge_metal");

#endif // MNN_METAL_ENABLED && MNN_LOW_MEMORY
