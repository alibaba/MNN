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
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static EXPRP _RoPEExpr(VARP q, VARP k, VARP cosEven, VARP cosOdd, VARP sinEven, VARP sinOdd, int ropeCutHeadDim) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_RoPE;
    op->main.type = OpParameter_Extra;
    op->main.value = new ExtraT;
    std::unique_ptr<AttributeT> attr(new AttributeT);
    attr->key = "rope_cut_head_dim";
    attr->i = ropeCutHeadDim;
    op->main.AsExtra()->attr.emplace_back(std::move(attr));
    return Expr::create(std::move(op), {q, k, cosEven, cosOdd, sinEven, sinOdd}, 2);
}

static void computeRopeExpected(const std::vector<float>& input, std::vector<float>& output,
                                const std::vector<float>& cosEven, const std::vector<float>& cosOdd,
                                const std::vector<float>& sinEven, const std::vector<float>& sinOdd, int outer,
                                int head, int headDim, int ropeCutHeadDim) {
    output = input;
    int halfDim = headDim / 2;
    int ropeHalfDim = std::min(ropeCutHeadDim / 2, halfDim);
    for (int o = 0; o < outer; ++o) {
        for (int h = 0; h < head; ++h) {
            for (int i = 0; i < ropeHalfDim; ++i) {
                int base = (o * head + h) * headDim;
                int trig = o * halfDim + i;
                float evenVal = input[base + i];
                float oddVal = input[base + i + halfDim];
                output[base + i] = evenVal * cosEven[trig] - oddVal * sinEven[trig];
                output[base + i + halfDim] = oddVal * cosOdd[trig] + evenVal * sinOdd[trig];
            }
        }
    }
}

class RoPETest : public MNNTestCase {
public:
    virtual ~RoPETest() = default;
    virtual bool run(int precision) {
        const int batch = 1;
        const int seqLen = 2;
        const int qHead = 2;
        const int kHead = 1;
        const int headDim = 6;
        const int halfDim = headDim / 2;
        const int outer = batch * seqLen;
        const int ropeCutHeadDim = 4;

        std::vector<float> qData = {
            0.2f, -0.4f, 0.6f, 1.0f, -1.2f, 1.4f,
            -0.7f, 0.9f, -1.1f, 0.3f, 0.5f, -0.8f,
            1.2f, -1.0f, 0.4f, -0.6f, 0.8f, -0.2f,
            -1.4f, 1.1f, -0.9f, 0.7f, -0.5f, 0.3f
        };
        std::vector<float> kData = {
            -0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f,
            1.0f, -1.1f, 1.2f, -1.3f, 1.4f, -1.5f
        };
        std::vector<float> cosEven = {0.90f, 0.80f, 0.70f, 0.60f, 0.50f, 0.40f};
        std::vector<float> cosOdd = {0.91f, 0.81f, 0.71f, 0.61f, 0.51f, 0.41f};
        std::vector<float> sinEven = {0.10f, -0.20f, 0.30f, -0.40f, 0.50f, -0.60f};
        std::vector<float> sinOdd = {0.11f, -0.21f, 0.31f, -0.41f, 0.51f, -0.61f};

        auto q = _Input({batch, seqLen, qHead, headDim}, NCHW);
        auto k = _Input({batch, seqLen, kHead, headDim}, NCHW);
        auto c0 = _Input({batch, seqLen, halfDim}, NCHW);
        auto c1 = _Input({batch, seqLen, halfDim}, NCHW);
        auto s0 = _Input({batch, seqLen, halfDim}, NCHW);
        auto s1 = _Input({batch, seqLen, halfDim}, NCHW);
        ::memcpy(q->writeMap<float>(), qData.data(), qData.size() * sizeof(float));
        ::memcpy(k->writeMap<float>(), kData.data(), kData.size() * sizeof(float));
        ::memcpy(c0->writeMap<float>(), cosEven.data(), cosEven.size() * sizeof(float));
        ::memcpy(c1->writeMap<float>(), cosOdd.data(), cosOdd.size() * sizeof(float));
        ::memcpy(s0->writeMap<float>(), sinEven.data(), sinEven.size() * sizeof(float));
        ::memcpy(s1->writeMap<float>(), sinOdd.data(), sinOdd.size() * sizeof(float));
        q->unMap();
        k->unMap();
        c0->unMap();
        c1->unMap();
        s0->unMap();
        s1->unMap();

        auto expr = _RoPEExpr(q, k, c0, c1, s0, s1, ropeCutHeadDim);
        auto qOut = Variable::create(expr, 0);
        auto kOut = Variable::create(expr, 1);
        std::vector<float> qExpected, kExpected;
        computeRopeExpected(qData, qExpected, cosEven, cosOdd, sinEven, sinOdd, outer, qHead, headDim, ropeCutHeadDim);
        computeRopeExpected(kData, kExpected, cosEven, cosOdd, sinEven, sinOdd, outer, kHead, headDim, ropeCutHeadDim);
        if (!checkVector<float>(qOut->readMap<float>(), qExpected.data(), qExpected.size(), 0.01f) ||
            !checkVector<float>(kOut->readMap<float>(), kExpected.data(), kExpected.size(), 0.01f)) {
            MNN_ERROR("RoPETest failed!\n");
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(RoPETest, "op/rope");

#endif
