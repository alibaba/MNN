//
//  RoPETest.cpp
//  MNNTests
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static EXPRP _RoPEExpr(VARP q, VARP k, VARP cos, VARP sin, int ropeCutHeadDim, int qHead, int kHead, int headDim,
                       const std::vector<float>* qNormGamma = nullptr, const std::vector<float>* kNormGamma = nullptr,
                       float normEps = 0.0f) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_RoPE;
    op->main.type = OpParameter_RoPEParam;
    op->main.value = new RoPEParamT;
    op->main.AsRoPEParam()->rope_cut_head_dim = ropeCutHeadDim;
    op->main.AsRoPEParam()->num_head = qHead;
    op->main.AsRoPEParam()->kv_num_head = kHead;
    op->main.AsRoPEParam()->head_dim = headDim;
    if (nullptr != qNormGamma) {
        op->main.AsRoPEParam()->q_norm.reset(new LayerNormT);
        op->main.AsRoPEParam()->q_norm->epsilon = normEps;
        op->main.AsRoPEParam()->q_norm->gamma = *qNormGamma;
        op->main.AsRoPEParam()->q_norm->useRMSNorm = true;
    }
    if (nullptr != kNormGamma) {
        op->main.AsRoPEParam()->k_norm.reset(new LayerNormT);
        op->main.AsRoPEParam()->k_norm->epsilon = normEps;
        op->main.AsRoPEParam()->k_norm->gamma = *kNormGamma;
        op->main.AsRoPEParam()->k_norm->useRMSNorm = true;
    }
    return Expr::create(std::move(op), {q, k, cos, sin}, 2);
}

static std::vector<float> packC4(const std::vector<float>& input, int seqLen, int channel) {
    std::vector<float> output(((channel + 3) / 4) * seqLen * 4, 0.0f);
    for (int t = 0; t < seqLen; ++t) {
        for (int c = 0; c < channel; ++c) {
            output[(c / 4) * seqLen * 4 + t * 4 + (c % 4)] = input[t * channel + c];
        }
    }
    return output;
}

static void computeRopeExpected(const std::vector<float>& input, std::vector<float>& output,
                                const std::vector<float>& cos, const std::vector<float>& sin, int outer, int head,
                                int headDim, int ropeCutHeadDim) {
    output = input;
    int ropeDim = std::min(ropeCutHeadDim, headDim);
    ropeDim = (ropeDim / 2) * 2;
    int ropeHalfDim = ropeDim / 2;
    for (int o = 0; o < outer; ++o) {
        for (int h = 0; h < head; ++h) {
            for (int i = 0; i < ropeHalfDim; ++i) {
                int base = (o * head + h) * headDim;
                int trig = o * ropeDim + i;
                float evenVal = input[base + i];
                float oddVal = input[base + i + ropeHalfDim];
                output[base + i] = evenVal * cos[trig] - oddVal * sin[trig];
                output[base + i + ropeHalfDim] = oddVal * cos[trig + ropeHalfDim] + evenVal * sin[trig + ropeHalfDim];
            }
        }
    }
}

static void computeRmsNormExpected(const std::vector<float>& input, std::vector<float>& output,
                                   const std::vector<float>& gamma, int outer, int head, int headDim, float eps) {
    output.resize(input.size());
    for (int o = 0; o < outer; ++o) {
        for (int h = 0; h < head; ++h) {
            int base = (o * head + h) * headDim;
            float sum = 0.0f;
            for (int i = 0; i < headDim; ++i) {
                float val = input[base + i];
                sum += val * val;
            }
            float scale = 1.0f / std::sqrt(sum / headDim + eps);
            for (int i = 0; i < headDim; ++i) {
                output[base + i] = input[base + i] * scale * gamma[i];
            }
        }
    }
}

class RoPETest : public MNNTestCase {
public:
    virtual ~RoPETest() = default;
    bool runCase(bool useNorm, int seqLen, int qHead, int kHead, int headDim, int ropeCutHeadDim) {
        const int batch = 1;
        const int outer = batch * seqLen;
        const int ropeHalfDim = ropeCutHeadDim / 2;
        const float normEps = 1e-6f;

        std::vector<float> qData(qHead * headDim * seqLen);
        std::vector<float> kData(kHead * headDim * seqLen);
        std::vector<float> cos(outer * ropeCutHeadDim);
        std::vector<float> sin(outer * ropeCutHeadDim);
        std::vector<float> qGamma(headDim);
        std::vector<float> kGamma(headDim);
        for (int i = 0; i < (int)qData.size(); ++i) {
            qData[i] = (float)((i % 13) - 6) * 0.17f;
        }
        for (int i = 0; i < (int)kData.size(); ++i) {
            kData[i] = (float)((i % 11) - 5) * -0.13f;
        }
        for (int i = 0; i < outer * ropeHalfDim; ++i) {
            int token = i / ropeHalfDim;
            int offset = i % ropeHalfDim;
            float angle = 0.013f * (token + 1) * (offset + 1);
            cos[token * ropeCutHeadDim + offset] = std::cos(angle);
            cos[token * ropeCutHeadDim + offset + ropeHalfDim] = std::cos(angle);
            sin[token * ropeCutHeadDim + offset] = std::sin(angle);
            sin[token * ropeCutHeadDim + offset + ropeHalfDim] = std::sin(angle);
        }
        for (int i = 0; i < headDim; ++i) {
            qGamma[i] = 0.7f + 0.03f * i;
            kGamma[i] = 1.2f - 0.04f * i;
        }
        auto qC4 = packC4(qData, seqLen, qHead * headDim);
        auto kC4 = packC4(kData, seqLen, kHead * headDim);

        auto q = _Input({seqLen, qHead * headDim, 1, 1}, NC4HW4);
        auto k = _Input({seqLen, kHead * headDim, 1, 1}, NC4HW4);
        auto c = _Input({batch, seqLen, ropeCutHeadDim}, NCHW);
        auto s = _Input({batch, seqLen, ropeCutHeadDim}, NCHW);
        ::memcpy(q->writeMap<float>(), qC4.data(), qC4.size() * sizeof(float));
        ::memcpy(k->writeMap<float>(), kC4.data(), kC4.size() * sizeof(float));
        ::memcpy(c->writeMap<float>(), cos.data(), cos.size() * sizeof(float));
        ::memcpy(s->writeMap<float>(), sin.data(), sin.size() * sizeof(float));
        q->unMap();
        k->unMap();
        c->unMap();
        s->unMap();

        auto expr = useNorm ? _RoPEExpr(q, k, c, s, ropeCutHeadDim, qHead, kHead, headDim, &qGamma, &kGamma, normEps)
                            : _RoPEExpr(q, k, c, s, ropeCutHeadDim, qHead, kHead, headDim);
        auto qOut = Variable::create(expr, 0);
        auto kOut = Variable::create(expr, 1);
        std::vector<float> qExpected, kExpected;
        if (useNorm) {
            std::vector<float> qNorm, kNorm;
            computeRmsNormExpected(qData, qNorm, qGamma, outer, qHead, headDim, normEps);
            computeRmsNormExpected(kData, kNorm, kGamma, outer, kHead, headDim, normEps);
            computeRopeExpected(qNorm, qExpected, cos, sin, outer, qHead, headDim, ropeCutHeadDim);
            computeRopeExpected(kNorm, kExpected, cos, sin, outer, kHead, headDim, ropeCutHeadDim);
        } else {
            computeRopeExpected(qData, qExpected, cos, sin, outer, qHead, headDim, ropeCutHeadDim);
            computeRopeExpected(kData, kExpected, cos, sin, outer, kHead, headDim, ropeCutHeadDim);
        }
        auto qExpectedC4 = packC4(qExpected, seqLen, qHead * headDim);
        auto kExpectedC4 = packC4(kExpected, seqLen, kHead * headDim);
        if (!checkVector<float>(qOut->readMap<float>(), qExpectedC4.data(), qExpectedC4.size(), 0.03f) ||
            !checkVector<float>(kOut->readMap<float>(), kExpectedC4.data(), kExpectedC4.size(), 0.03f)) {
            MNN_ERROR("RoPETest %s failed!\n", useNorm ? "norm" : "base");
            return false;
        }
        return true;
    }

    virtual bool run(int precision) {
        return runCase(false, 2, 2, 1, 8, 6) && runCase(true, 2, 2, 1, 8, 6) && runCase(false, 18, 16, 2, 128, 128) &&
               runCase(true, 18, 16, 2, 128, 128);
    }
};

MNNTestSuiteRegister(RoPETest, "op/rope");

#endif
