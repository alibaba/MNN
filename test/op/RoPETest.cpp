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
                       float normEps = 0.0f, bool interleaved = false, bool normAfter = false) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_RoPE;
    op->main.type = OpParameter_RoPEParam;
    op->main.value = new RoPEParamT;
    op->main.AsRoPEParam()->rope_cut_head_dim = ropeCutHeadDim;
    op->main.AsRoPEParam()->num_head = qHead;
    op->main.AsRoPEParam()->kv_num_head = kHead;
    op->main.AsRoPEParam()->head_dim = headDim;
    op->main.AsRoPEParam()->interleaved = interleaved;
    op->main.AsRoPEParam()->norm_after = normAfter;
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

static EXPRP _RoPEExprInterleaved(VARP q, VARP k, VARP cos, VARP sin, int ropeCutHeadDim, int qHead, int kHead,
                                  int headDim, float qkNormEps = 0.0f) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_RoPE;
    op->main.type = OpParameter_RoPEParam;
    op->main.value = new RoPEParamT;
    op->main.AsRoPEParam()->rope_cut_head_dim = ropeCutHeadDim;
    op->main.AsRoPEParam()->num_head = qHead;
    op->main.AsRoPEParam()->kv_num_head = kHead;
    op->main.AsRoPEParam()->head_dim = headDim;
    op->main.AsRoPEParam()->interleaved = true;
    if (qkNormEps > 0.0f) {
        op->main.AsRoPEParam()->q_norm.reset(new LayerNormT);
        op->main.AsRoPEParam()->q_norm->epsilon = qkNormEps;
        op->main.AsRoPEParam()->q_norm->useRMSNorm = true;
        op->main.AsRoPEParam()->k_norm.reset(new LayerNormT);
        op->main.AsRoPEParam()->k_norm->epsilon = qkNormEps;
        op->main.AsRoPEParam()->k_norm->useRMSNorm = true;
    }
    return Expr::create(std::move(op), {q, k, cos, sin}, 2);
}

static EXPRP _RoPEExprGammaFreeNorm(VARP q, VARP k, VARP cos, VARP sin, int ropeCutHeadDim, int qHead, int kHead,
                                    int headDim, float normEps) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_RoPE;
    op->main.type = OpParameter_RoPEParam;
    op->main.value = new RoPEParamT;
    op->main.AsRoPEParam()->rope_cut_head_dim = ropeCutHeadDim;
    op->main.AsRoPEParam()->num_head = qHead;
    op->main.AsRoPEParam()->kv_num_head = kHead;
    op->main.AsRoPEParam()->head_dim = headDim;
    op->main.AsRoPEParam()->q_norm.reset(new LayerNormT);
    op->main.AsRoPEParam()->q_norm->epsilon = normEps;
    op->main.AsRoPEParam()->q_norm->useRMSNorm = true;
    op->main.AsRoPEParam()->k_norm.reset(new LayerNormT);
    op->main.AsRoPEParam()->k_norm->epsilon = normEps;
    op->main.AsRoPEParam()->k_norm->useRMSNorm = true;
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
    int ropeDim = ropeCutHeadDim <= 0 ? headDim : std::min(ropeCutHeadDim, headDim);
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

static void computeRopeInterleavedExpected(const std::vector<float>& input, std::vector<float>& output,
                                           const std::vector<float>& cos, const std::vector<float>& sin, int outer,
                                           int head, int headDim, int ropeCutHeadDim) {
    output = input;
    int ropeDim = ropeCutHeadDim <= 0 ? headDim : std::min(ropeCutHeadDim, headDim);
    ropeDim = (ropeDim / 2) * 2;
    int ropeHalfDim = ropeDim / 2;
    for (int o = 0; o < outer; ++o) {
        for (int h = 0; h < head; ++h) {
            int base = (o * head + h) * headDim;
            for (int i = 0; i < ropeHalfDim; ++i) {
                int trig = o * ropeDim + 2 * i;
                float evenVal = input[base + 2 * i];
                float oddVal = input[base + 2 * i + 1];
                output[base + 2 * i] = evenVal * cos[trig] - oddVal * sin[trig];
                output[base + 2 * i + 1] = evenVal * sin[trig] + oddVal * cos[trig];
            }
        }
    }
}

static void computeRmsNormGammaFreeExpected(const std::vector<float>& input, std::vector<float>& output, int outer,
                                            int head, int headDim, float eps) {
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
                output[base + i] = input[base + i] * scale;
            }
        }
    }
}

class RoPETest : public MNNTestCase {
public:
    virtual ~RoPETest() = default;
    bool runCase(bool useNorm, int seqLen, int qHead, int kHead, int headDim, int ropeCutHeadDim,
                 bool interleaved = false, int normMask = 3, bool gammaFree = false) {
        const int batch = 1;
        const int outer = batch * seqLen;
        int ropeDim = ropeCutHeadDim <= 0 ? headDim : std::min(ropeCutHeadDim, headDim);
        ropeDim = (ropeDim / 2) * 2;
        const int ropeHalfDim = ropeDim / 2;
        const int trigWidth = std::max(ropeDim, 1);
        const float normEps = 1e-6f;

        std::vector<float> qData(qHead * headDim * seqLen);
        std::vector<float> kData(kHead * headDim * seqLen);
        std::vector<float> cos(outer * trigWidth);
        std::vector<float> sin(outer * trigWidth);
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
            const int even = interleaved ? 2 * offset : offset;
            const int odd = interleaved ? even + 1 : offset + ropeHalfDim;
            cos[token * trigWidth + even] = cos[token * trigWidth + odd] = std::cos(angle);
            sin[token * trigWidth + even] = sin[token * trigWidth + odd] = std::sin(angle);
        }
        for (int i = 0; i < headDim; ++i) {
            qGamma[i] = 0.7f + 0.03f * i;
            kGamma[i] = 1.2f - 0.04f * i;
        }
        auto qC4 = packC4(qData, seqLen, qHead * headDim);
        auto kC4 = packC4(kData, seqLen, kHead * headDim);

        auto q = _Input({seqLen, qHead * headDim, 1, 1}, NC4HW4);
        auto k = _Input({seqLen, kHead * headDim, 1, 1}, NC4HW4);
        auto c = _Input({batch, seqLen, trigWidth}, NCHW);
        auto s = _Input({batch, seqLen, trigWidth}, NCHW);
        ::memcpy(q->writeMap<float>(), qC4.data(), qC4.size() * sizeof(float));
        ::memcpy(k->writeMap<float>(), kC4.data(), kC4.size() * sizeof(float));
        ::memcpy(c->writeMap<float>(), cos.data(), cos.size() * sizeof(float));
        ::memcpy(s->writeMap<float>(), sin.data(), sin.size() * sizeof(float));
        q->unMap();
        k->unMap();
        c->unMap();
        s->unMap();

        const bool qNorm = useNorm && (normMask & 1);
        const bool kNorm = useNorm && (normMask & 2);
        std::vector<float> emptyGamma;
        auto expr = _RoPEExpr(q, k, c, s, ropeCutHeadDim, qHead, kHead, headDim,
                             qNorm ? (gammaFree ? &emptyGamma : &qGamma) : nullptr,
                             kNorm ? (gammaFree ? &emptyGamma : &kGamma) : nullptr, normEps, interleaved);
        auto qOut = Variable::create(expr, 0);
        auto kOut = Variable::create(expr, 1);
        std::vector<float> qExpected, kExpected, qIn = qData, kIn = kData;
        auto rotate = interleaved ? computeRopeInterleavedExpected : computeRopeExpected;
        if (qNorm) {
            if (gammaFree) {
                computeRmsNormGammaFreeExpected(qData, qIn, outer, qHead, headDim, normEps);
            } else {
                computeRmsNormExpected(qData, qIn, qGamma, outer, qHead, headDim, normEps);
            }
        }
        if (kNorm) {
            if (gammaFree) {
                computeRmsNormGammaFreeExpected(kData, kIn, outer, kHead, headDim, normEps);
            } else {
                computeRmsNormExpected(kData, kIn, kGamma, outer, kHead, headDim, normEps);
            }
        }
        rotate(qIn, qExpected, cos, sin, outer, qHead, headDim, ropeCutHeadDim);
        rotate(kIn, kExpected, cos, sin, outer, kHead, headDim, ropeCutHeadDim);
        if (!checkVector<float>(qOut->readMap<float>(), qExpected.data(), qExpected.size(), 0.03f) ||
            !checkVector<float>(kOut->readMap<float>(), kExpected.data(), kExpected.size(), 0.03f)) {
            MNN_ERROR("RoPETest failed: norm=%d seq=%d qHead=%d kHead=%d headDim=%d cut=%d interleaved=%d "
                      "normMask=%d gammaFree=%d\n",
                      (int)useNorm, seqLen, qHead, kHead, headDim, ropeCutHeadDim, (int)interleaved, normMask,
                      (int)gammaFree);
            return false;
        }
        return true;
    }

    virtual bool run(int precision) {
        bool base = runCase(false, 2, 2, 1, 8, 6) && runCase(true, 2, 2, 1, 8, 6) &&
                    runCase(false, 18, 16, 2, 128, 128) && runCase(true, 18, 16, 2, 128, 128) &&
                    runCase(true, 3, 8, 2, 256, 64);
        bool inter = runCaseInterleaved(false, 2, 2, 1, 8, 8) && runCaseInterleaved(true, 2, 2, 1, 8, 8) &&
                     runCaseInterleaved(true, 3, 12, 4, 64, 64) && runCaseInterleaved(true, 5, 12, 4, 64, 32);
        bool hsGf = runCaseHalfSplitGammaFree(2, 2, 1, 8, 8) && runCaseHalfSplitGammaFree(3, 12, 4, 64, 64) &&
                    runCaseHalfSplitGammaFree(5, 12, 4, 64, 32) && runCaseHalfSplitGammaFree(96, 12, 4, 64, 64);
        bool interGamma = runCase(true, 1, 12, 4, 64, 64, true) &&
                          runCase(true, 37, 12, 4, 64, 32, true) &&
                          runCase(true, 3, 2, 1, 8, 6, true) &&
                          runCase(false, 3, 2, 1, 10, 6, true);
        bool blocks = true;
        for (bool interleaved : {false, true}) {
            for (int headDim : {6, 12, 16, 20, 24, 30, 32, 34, 40, 64, 96, 128}) {
                for (int cut : {headDim, headDim - 2, headDim / 2}) {
                    blocks &= runCase(false, 3, 3, 3, headDim, cut, interleaved);
                }
            }
        }
        bool edges = true;
        for (bool interleaved : {false, true}) {
            for (int cut : {0, -1, 7, 35, 68}) {
                edges &= runCase(false, 2, 5, 3, 64, cut, interleaved);
            }
            edges &= runCase(true, 3, 3, 1, 64, 64, interleaved);
            edges &= runCase(false, 2, 3, 3, 10, 7, interleaved);
        }
        for (int normMask : {1, 2}) {
            for (bool gammaFree : {false, true}) {
                for (bool interleaved : {false, true}) {
                    for (int seqLen : {1, 96}) {
                        for (int headDim : {8, 64}) {
                            if (!runCase(true, seqLen, 2, 1, headDim, headDim, interleaved, normMask, gammaFree)) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        bool post = true;
        for (bool interleaved : {false, true}) {
            for (bool gammaFree : {false, true}) {
                for (float scale : {1.0f, 1.3f}) {
                    for (int seqLen : {1, 96}) {
                        post &= runCasePostNorm(seqLen, 2, 1, 64, 64, interleaved, gammaFree, scale);
                        post &= runCasePostNorm(seqLen, 12, 4, 64, 32, interleaved, gammaFree, scale);
                    }
                }
            }
        }
        bool pre = true;
        for (bool interleaved : {false, true}) {
            for (bool gammaFree : {false, true}) {
                for (int cut : {64, 32}) {
                    pre &= runCasePostNorm(3, 2, 1, 64, cut, interleaved, gammaFree, 1.0f, false);
                }
            }
        }
        bool inplace = true;
        for (bool interleaved : {false, true}) {
            for (int cut : {32, 30, 16}) {
                inplace &= runCase(true, 5, 3, 2, 64, cut, interleaved);
            }
        }
        return base && inter && hsGf && interGamma && blocks && edges && post && pre && inplace;
    }

    bool runCaseHalfSplitGammaFree(int seqLen, int qHead, int kHead, int headDim, int ropeCutHeadDim) {
        const int batch = 1;
        const int outer = batch * seqLen;
        const int ropeHalfDim = ropeCutHeadDim / 2;
        const float normEps = 1e-5f;

        std::vector<float> qData(qHead * headDim * seqLen);
        std::vector<float> kData(kHead * headDim * seqLen);
        std::vector<float> cos(outer * ropeCutHeadDim);
        std::vector<float> sin(outer * ropeCutHeadDim);
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

        auto expr = _RoPEExprGammaFreeNorm(q, k, c, s, ropeCutHeadDim, qHead, kHead, headDim, normEps);
        auto qOut = Variable::create(expr, 0);
        auto kOut = Variable::create(expr, 1);
        std::vector<float> qNorm, kNorm, qExpected, kExpected;
        computeRmsNormGammaFreeExpected(qData, qNorm, outer, qHead, headDim, normEps);
        computeRmsNormGammaFreeExpected(kData, kNorm, outer, kHead, headDim, normEps);
        computeRopeExpected(qNorm, qExpected, cos, sin, outer, qHead, headDim, ropeCutHeadDim);
        computeRopeExpected(kNorm, kExpected, cos, sin, outer, kHead, headDim, ropeCutHeadDim);
        if (!checkVector<float>(qOut->readMap<float>(), qExpected.data(), qExpected.size(), 0.02f) ||
            !checkVector<float>(kOut->readMap<float>(), kExpected.data(), kExpected.size(), 0.02f)) {
            MNN_ERROR("RoPETest half-split gamma-free norm failed!\n");
            return false;
        }
        return true;
    }

    bool runCaseInterleaved(bool useNorm, int seqLen, int qHead, int kHead, int headDim, int ropeCutHeadDim) {
        const int batch = 1;
        const int outer = batch * seqLen;
        const int ropeDim = (std::min(ropeCutHeadDim, headDim) / 2) * 2;
        const int ropeHalfDim = ropeDim / 2;
        const float normEps = 1e-5f;

        std::vector<float> qData(qHead * headDim * seqLen);
        std::vector<float> kData(kHead * headDim * seqLen);
        std::vector<float> cos(outer * ropeDim);
        std::vector<float> sin(outer * ropeDim);
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
            cos[token * ropeDim + 2 * offset] = std::cos(angle);
            cos[token * ropeDim + 2 * offset + 1] = std::cos(angle);
            sin[token * ropeDim + 2 * offset] = std::sin(angle);
            sin[token * ropeDim + 2 * offset + 1] = std::sin(angle);
        }
        auto qC4 = packC4(qData, seqLen, qHead * headDim);
        auto kC4 = packC4(kData, seqLen, kHead * headDim);
        auto q = _Input({seqLen, qHead * headDim, 1, 1}, NC4HW4);
        auto k = _Input({seqLen, kHead * headDim, 1, 1}, NC4HW4);
        auto c = _Input({batch, seqLen, ropeDim}, NCHW);
        auto s = _Input({batch, seqLen, ropeDim}, NCHW);
        ::memcpy(q->writeMap<float>(), qC4.data(), qC4.size() * sizeof(float));
        ::memcpy(k->writeMap<float>(), kC4.data(), kC4.size() * sizeof(float));
        ::memcpy(c->writeMap<float>(), cos.data(), cos.size() * sizeof(float));
        ::memcpy(s->writeMap<float>(), sin.data(), sin.size() * sizeof(float));
        q->unMap();
        k->unMap();
        c->unMap();
        s->unMap();

        auto expr = _RoPEExprInterleaved(q, k, c, s, ropeCutHeadDim, qHead, kHead, headDim, useNorm ? normEps : 0.0f);
        auto qOut = Variable::create(expr, 0);
        auto kOut = Variable::create(expr, 1);
        std::vector<float> qExpected, kExpected, qIn, kIn;
        if (useNorm) {
            computeRmsNormGammaFreeExpected(qData, qIn, outer, qHead, headDim, normEps);
            computeRmsNormGammaFreeExpected(kData, kIn, outer, kHead, headDim, normEps);
        } else {
            qIn = qData;
            kIn = kData;
        }
        computeRopeInterleavedExpected(qIn, qExpected, cos, sin, outer, qHead, headDim, ropeCutHeadDim);
        computeRopeInterleavedExpected(kIn, kExpected, cos, sin, outer, kHead, headDim, ropeCutHeadDim);
        if (!checkVector<float>(qOut->readMap<float>(), qExpected.data(), qExpected.size(), 0.02f) ||
            !checkVector<float>(kOut->readMap<float>(), kExpected.data(), kExpected.size(), 0.02f)) {
            MNN_ERROR("RoPETest interleaved %s failed!\n", useNorm ? "norm" : "base");
            return false;
        }
        return true;
    }

    bool runCasePostNorm(int seqLen, int qHead, int kHead, int headDim, int ropeCutHeadDim, bool interleaved,
                         bool gammaFree, float tableScale, bool normAfter = true) {
        const int batch = 1;
        const int outer = batch * seqLen;
        const int ropeDim = (std::min(ropeCutHeadDim, headDim) / 2) * 2;
        const int ropeHalfDim = ropeDim / 2;
        const float normEps = 1e-5f;

        std::vector<float> qData(qHead * headDim * seqLen);
        std::vector<float> kData(kHead * headDim * seqLen);
        std::vector<float> cos(outer * ropeDim);
        std::vector<float> sin(outer * ropeDim);
        std::vector<float> qGamma(headDim), kGamma(headDim);
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
            const int even = interleaved ? 2 * offset : offset;
            const int odd = interleaved ? even + 1 : offset + ropeHalfDim;
            cos[token * ropeDim + even] = cos[token * ropeDim + odd] = tableScale * std::cos(angle);
            sin[token * ropeDim + even] = sin[token * ropeDim + odd] = tableScale * std::sin(angle);
        }
        for (int i = 0; i < headDim; ++i) {
            qGamma[i] = 0.7f + 0.03f * i;
            kGamma[i] = 1.2f - 0.04f * i;
        }
        auto qC4 = packC4(qData, seqLen, qHead * headDim);
        auto kC4 = packC4(kData, seqLen, kHead * headDim);
        auto q = _Input({seqLen, qHead * headDim, 1, 1}, NC4HW4);
        auto k = _Input({seqLen, kHead * headDim, 1, 1}, NC4HW4);
        auto c = _Input({batch, seqLen, ropeDim}, NCHW);
        auto s = _Input({batch, seqLen, ropeDim}, NCHW);
        ::memcpy(q->writeMap<float>(), qC4.data(), qC4.size() * sizeof(float));
        ::memcpy(k->writeMap<float>(), kC4.data(), kC4.size() * sizeof(float));
        ::memcpy(c->writeMap<float>(), cos.data(), cos.size() * sizeof(float));
        ::memcpy(s->writeMap<float>(), sin.data(), sin.size() * sizeof(float));
        q->unMap();
        k->unMap();
        c->unMap();
        s->unMap();

        std::vector<float> emptyGamma;
        auto expr = _RoPEExpr(q, k, c, s, ropeCutHeadDim, qHead, kHead, headDim,
                             gammaFree ? &emptyGamma : &qGamma, gammaFree ? &emptyGamma : &kGamma, normEps, interleaved,
                             normAfter);
        auto qOut = Variable::create(expr, 0);
        auto kOut = Variable::create(expr, 1);
        auto rotate = interleaved ? computeRopeInterleavedExpected : computeRopeExpected;
        auto norm = [&](const std::vector<float>& in, std::vector<float>& out, const std::vector<float>& gamma,
                        int heads) {
            if (gammaFree) {
                computeRmsNormGammaFreeExpected(in, out, outer, heads, headDim, normEps);
            } else {
                computeRmsNormExpected(in, out, gamma, outer, heads, headDim, normEps);
            }
        };
        std::vector<float> qRot, kRot, qPre, kPre, qExpected, kExpected;
        rotate(qData, qRot, cos, sin, outer, qHead, headDim, ropeCutHeadDim);
        rotate(kData, kRot, cos, sin, outer, kHead, headDim, ropeCutHeadDim);
        if (normAfter) {
            norm(qRot, qExpected, qGamma, qHead);
        } else {
            norm(qData, qPre, qGamma, qHead);
            rotate(qPre, qExpected, cos, sin, outer, qHead, headDim, ropeCutHeadDim);
        }
        if (normAfter) {
            norm(kRot, kExpected, kGamma, kHead);
        } else {
            norm(kData, kPre, kGamma, kHead);
            rotate(kPre, kExpected, cos, sin, outer, kHead, headDim, ropeCutHeadDim);
        }
        if (!checkVector<float>(qOut->readMap<float>(), qExpected.data(), qExpected.size(), 0.02f) ||
            !checkVector<float>(kOut->readMap<float>(), kExpected.data(), kExpected.size(), 0.02f)) {
            MNN_ERROR("RoPETest post-norm (interleaved=%d gammaFree=%d scale=%.2f normAfter=%d) failed!\n",
                      interleaved, gammaFree, tableScale, normAfter);
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(RoPETest, "op/rope");

#endif
