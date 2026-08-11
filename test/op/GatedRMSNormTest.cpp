//
//  GatedRMSNormTest.cpp
//  MNNTests
//
//  Tests for OpType_GatedRMSNorm: out = RMSNorm(x) * silu(z).
//
//  The op absorbs the C4 repacks that used to bracket the chain, so its inputs
//  carry different layouts on purpose: x is [outside, inside] with the head as
//  the batch axis, while z and the output are [1, outside*inside]. A layout
//  mix-up shows up as a value mismatch here.
//
//  CPU exercises the geometry decomposition; Metal exercises the fused kernel.
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
    // batch == 1 is the decode case (fused Metal kernel); batch > 1 is prefill,
    // which decomposes on every backend.
    static bool runCase(int batch, int heads, int inside) {
        const int outside = batch * heads;
        const int total   = outside * inside;
        const float eps   = 1e-6f;

        std::vector<float> xData(total), zData(total), gamma(inside), beta(inside);
        for (int i = 0; i < total; ++i) {
            xData[i] = (float)((i % 11) - 5) * 0.17f;
            zData[i] = (float)((i % 7) - 3) * 0.29f;
        }
        for (int c = 0; c < inside; ++c) {
            gamma[c] = 0.9f + 0.03f * c;
            beta[c]  = 0.01f * c - 0.02f;
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
        if (!checkVector<float>(out->readMap<float>(), expectedC4.data(), expectedC4.size(), 0.005f)) {
            MNN_ERROR("GatedRMSNorm(batch=%d): value mismatch\n", batch);
            return false;
        }
        return true;
    }

public:
    virtual ~GatedRMSNormTest() = default;
    virtual bool run(int precision) {
        // decode: batch 1, fused Metal kernel; prefill: batch 3, decomposed.
        // inside=6 is not 4-aligned, so the fused kernel cannot take it and
        // Metal must decompose as well; the batch>1 variant also checks that
        // the decomposition's flat view of the normalized result does not pick
        // up NC4HW4 channel padding.
        return runCase(1, 4, 8) && runCase(3, 4, 8) && runCase(1, 4, 6) && runCase(2, 3, 6);
    }
};
MNNTestSuiteRegister(GatedRMSNormTest, "op/gated_rms_norm");

#endif
