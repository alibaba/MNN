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
    int halfDim = headDim / 2;
    int ropeHalfDim = std::min(ropeCutHeadDim / 2, halfDim);
    for (int o = 0; o < outer; ++o) {
        for (int h = 0; h < head; ++h) {
            for (int i = 0; i < ropeHalfDim; ++i) {
                int base = (o * head + h) * headDim;
                int trig = o * headDim + i;
                float evenVal = input[base + i];
                float oddVal = input[base + i + halfDim];
                output[base + i] = evenVal * cos[trig] - oddVal * sin[trig];
                output[base + i + halfDim] = oddVal * cos[trig + halfDim] + evenVal * sin[trig + halfDim];
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
    bool runCase(bool useNorm) {
        const int batch = 1;
        const int seqLen = 2;
        const int qHead = 2;
        const int kHead = 1;
        const int headDim = 8;
        const int halfDim = headDim / 2;
        const int outer = batch * seqLen;
        const int ropeCutHeadDim = 6;
        const float normEps = 1e-6f;

        std::vector<float> qData(qHead * headDim * seqLen);
        std::vector<float> kData(kHead * headDim * seqLen);
        std::vector<float> cos(outer * headDim);
        std::vector<float> sin(outer * headDim);
        std::vector<float> qGamma(headDim);
        std::vector<float> kGamma(headDim);
        for (int i = 0; i < (int)qData.size(); ++i) {
            qData[i] = (float)((i % 13) - 6) * 0.17f;
        }
        for (int i = 0; i < (int)kData.size(); ++i) {
            kData[i] = (float)((i % 11) - 5) * -0.13f;
        }
        for (int i = 0; i < outer * halfDim; ++i) {
            int token = i / halfDim;
            int offset = i % halfDim;
            cos[token * headDim + offset] = 0.9f - 0.03f * i;
            cos[token * headDim + offset + halfDim] = 0.91f - 0.02f * i;
            sin[token * headDim + offset] = 0.1f + 0.04f * i;
            sin[token * headDim + offset + halfDim] = 0.11f + 0.03f * i;
        }
        for (int i = 0; i < headDim; ++i) {
            qGamma[i] = 0.7f + 0.03f * i;
            kGamma[i] = 1.2f - 0.04f * i;
        }
        auto qC4 = packC4(qData, seqLen, qHead * headDim);
        auto kC4 = packC4(kData, seqLen, kHead * headDim);

        auto q = _Input({seqLen, qHead * headDim, 1, 1}, NC4HW4);
        auto k = _Input({seqLen, kHead * headDim, 1, 1}, NC4HW4);
        auto c = _Input({batch, seqLen, headDim}, NCHW);
        auto s = _Input({batch, seqLen, headDim}, NCHW);
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
        if (!checkVector<float>(qOut->readMap<float>(), qExpected.data(), qExpected.size(), 0.03f) ||
            !checkVector<float>(kOut->readMap<float>(), kExpected.data(), kExpected.size(), 0.03f)) {
            MNN_ERROR("RoPETest %s failed!\n", useNorm ? "norm" : "base");
            return false;
        }
        return true;
    }

    virtual bool run(int precision) {
        return runCase(false) && runCase(true);
    }
};

MNNTestSuiteRegister(RoPETest, "op/rope");

#endif
