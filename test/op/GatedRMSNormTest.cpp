//
//  GatedRMSNormTest.cpp
//  MNNTests
//
//  Tests for OpType_GatedRMSNorm: out = RMSNorm(x) * silu(z).
//
//  The op absorbs the C4 repacks that used to bracket the chain, so its inputs
//  carry different layouts on purpose: x is [batch*heads, inside] with the head
//  folded into the batch axis, while z and the output are [batch, heads*inside].
//  A layout mix-up shows up as a value mismatch here.
//
//  CPU exercises the geometry decomposition; Metal exercises the fused kernel
//  whenever the shape contract holds, for prefill as well as decode.
//

#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && defined(MNN_GATED_RMS_NORM)

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cmath>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

// plain [seq, channel] -> NC4HW4
static std::vector<float> packC4Gn(const std::vector<float>& input, int seqLen, int channel) {
    std::vector<float> output(((channel + 3) / 4) * seqLen * 4, 0.0f);
    for (int t = 0; t < seqLen; ++t) {
        for (int c = 0; c < channel; ++c) {
            output[(c / 4) * seqLen * 4 + t * 4 + (c % 4)] = input[t * channel + c];
        }
    }
    return output;
}

class GatedRMSNormTest : public MNNTestCase {
    // x: [batch*heads, inside] with head as batch axis; z/out: [batch, heads*inside].
    // batch == 1 is decode, batch > 1 prefill; Metal fuses both when inside is
    // 4-aligned, and decomposes otherwise.
    static bool runCase(int batch, int heads, int inside) {
        const int outside = batch * heads;
        const int total   = outside * inside;
        const float eps   = 1e-6f;

        std::vector<float> xData(total), zData(total), gamma(inside), beta(inside);
        for (int i = 0; i < total; ++i) {
            xData[i] = (float)((i % 11) - 5) * 0.17f;
            zData[i] = (float)((i % 7) - 3) * 0.29f;
        }
        // Keep gamma / beta near the range a real RMSNorm carries. They still vary with every channel, so
        // a channel mix-up is caught, but they no longer scale the output with `inside`: at 0.03 per
        // channel the inside=256 case peaked at ~11, where fp16's ~5e-4 relative error alone exceeds the
        // absolute tolerance below.
        for (int c = 0; c < inside; ++c) {
            gamma[c] = 0.9f + 0.003f * c;
            beta[c] = 0.001f * c - 0.02f;
        }

        auto x = _Input({outside, inside, 1, 1}, NC4HW4);
        auto xC4 = packC4Gn(xData, outside, inside);
        ::memcpy(x->writeMap<float>(), xC4.data(), xC4.size() * sizeof(float));
        x->unMap();

        auto z = _Input({batch, heads * inside, 1, 1}, NC4HW4);
        auto zC4 = packC4Gn(zData, batch, heads * inside);
        ::memcpy(z->writeMap<float>(), zC4.data(), zC4.size() * sizeof(float));
        z->unMap();

        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_GatedRMSNorm;
        op->main.type = OpParameter_LayerNorm;
        op->main.value = new LayerNormT;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto param = op->main.AsLayerNorm();
        param->epsilon    = eps;
        param->gamma      = gamma;
        param->beta       = beta;
        param->axis       = {-1};
        param->useRMSNorm = true;

        auto out = Variable::create(Expr::create(std::move(op), {x, z}, 1));

        // Reference: per row h of x, rms-normalize, scale by gamma/beta, then
        // multiply by silu(z) at the same flat element index.
        std::vector<float> expected(total);
        for (int h = 0; h < outside; ++h) {
            float ss = 0.0f;
            for (int c = 0; c < inside; ++c) {
                float v = xData[h * inside + c];
                ss += v * v;
            }
            float inv = 1.0f / sqrtf(ss / inside + eps);
            for (int c = 0; c < inside; ++c) {
                float normed = xData[h * inside + c] * inv * gamma[c] + beta[c];
                float g      = zData[h * inside + c];
                float silu   = g / (1.0f + expf(-g));
                expected[h * inside + c] = normed * silu;
            }
        }

        auto info = out->getInfo();
        if (info == nullptr || info->dim.size() != 4 || info->dim[0] != batch ||
            info->dim[1] != heads * inside) {
            MNN_ERROR("GatedRMSNorm(batch=%d): bad output shape\n", batch);
            return false;
        }
        auto expectedC4 = packC4Gn(expected, batch, heads * inside);
        // Hexagon computes in fp16 and runs silu through an fp16 PWL, which costs it ~0.5% relative --
        // five times what the CPU's fp16 path carries -- so it gets a relative bound.
        const bool ok = getCurrentType() == MNN_FORWARD_HEXAGON
                            ? checkVectorByRelativeError<float>(out->readMap<float>(), expectedC4.data(),
                                                                (int)expectedC4.size(), 0.01f)
                            : checkVector<float>(out->readMap<float>(), expectedC4.data(), expectedC4.size(), 0.005f);
        if (!ok) {
            MNN_ERROR("GatedRMSNorm(batch=%d): value mismatch\n", batch);
            return false;
        }
        return true;
    }

public:
    virtual ~GatedRMSNormTest() = default;
    virtual bool run(int precision) {
        // batch 1 is decode and batch 3 / 5 prefill; both fuse on Metal.
        // inside=6 is not 4-aligned, so the fused kernel cannot take it and
        // Metal must decompose — that variant also checks that the
        // decomposition's flat view of the normalized result does not pick up
        // NC4HW4 channel padding. inside=128 pushes the fused reduction past
        // one iteration per lane.
        return runCase(1, 4, 8) && runCase(3, 4, 8) && runCase(1, 4, 6) && runCase(2, 3, 6) && runCase(5, 3, 128) &&
               runCase(1, 3, 128) &&
               // Hexagon fuses inside % 64 == 0 and splits prefill into 64-token units across workers:
               // 70 tokens leave a partial unit, inside=256 gives several channel blocks per row.
               runCase(70, 16, 128) && runCase(130, 2, 256);
    }
};
MNNTestSuiteRegister(GatedRMSNormTest, "op/gated_rms_norm");

#endif
