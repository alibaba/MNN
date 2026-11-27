//
//  ArgMaxTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class ArgMaxTest : public MNNTestCase {
public:
    virtual ~ArgMaxTest() = default;
    virtual bool run(int precision) {
        auto ArgMax_ = [](VARP input, int axis, int topK, int outMaxVal) {
            using namespace MNN;
            // input = _checkNC4HW4(input);
            std::unique_ptr<OpT> op(new OpT);
            op->main.type                         = OpParameter_ArgMax;
            op->type                              = OpType_ArgMax;
            op->main.value                        = new ArgMaxT;
            op->main.AsArgMax()->axis = axis;
            op->main.AsArgMax()->outMaxVal = outMaxVal;
            op->main.AsArgMax()->topK = topK;
            op->main.AsArgMax()->softmaxThreshold = 0;
            return (Variable::create(Expr::create(std::move(op), {input})));
        };
        // auto input_nhwc = _Input({128 * 1600, 64}, NHWC);
        auto input_nhwc = _Input({4, 4}, NHWC);
        auto input_nchw = _Input({4, 4}, NC4HW4);
        input_nhwc->setName("input_tensor_nhwc");
        input_nchw->setName("input_tensor_nchw");
        // set input data
        const float inpudata[] = {-1.0, 2.0,   -3.0, 4.0,
                                  5.0,  -6.0, 7.0,   -8.0,
                                  -9.0, -10.0, 11.0, 12.0,
                                  13.0, 14.0, -15.0, -16.0};
        auto inputPtr          = input_nhwc->writeMap<float>();
        memset(inputPtr, 0, input_nhwc->getInfo()->size * sizeof(float));
        memcpy(inputPtr, inpudata, 16 * sizeof(float));
        inputPtr          = input_nchw->writeMap<float>();
        memcpy(inputPtr, inpudata, 16 * sizeof(float));
        input_nhwc->unMap();
        input_nchw->unMap();
        auto output_0                           = _ArgMax(input_nhwc, 0);
        auto output_1                           = _ArgMax(input_nhwc, 1);
        auto output_2                           = ArgMax_(input_nchw, 1, 2, 0);
        auto output_3                           = ArgMax_(input_nchw, 1, 1, 1);
        const std::vector<int> expectedOutput_0 = {3, 3, 2, 2};
        const std::vector<int> expectedOutput_1 = {3, 2, 3, 1};
        const std::vector<float> expectedOutput_2 = {3, 1, 2, 0, 3, 2, 1, 0};
        const std::vector<float> expectedOutput_3 = {3, 4, 2, 7, 3, 12, 1, 14};
        auto gotOutput_0                        = output_0->readMap<int>();
        auto gotOutput_1                        = output_1->readMap<int>();
        auto gotOutput_2                        = output_2->readMap<float>();
        auto gotOutput_3                        = output_3->readMap<float>();
        if (!checkVector<int>(gotOutput_0, expectedOutput_0.data(), 4, 0)) {
            MNN_ERROR("ArgMaxTest test axis_0 failed!\n");
            return false;
        }
        if (!checkVector<int>(gotOutput_1, expectedOutput_1.data(), 4, 0)) {
            MNN_ERROR("ArgMaxTest test axis_1 failed!\n");
            return false;
        }
        if (!checkVector<float>(gotOutput_2, expectedOutput_2.data(), 8, 0)) {
            MNN_ERROR("ArgMaxTest test axis_1_top2 failed!\n");
            return false;
        }
        if (!checkVector<float>(gotOutput_3, expectedOutput_3.data(), 8, 0)) {
            MNN_ERROR("ArgMaxTest test axis_1_outVal failed!\n");
            return false;
        }
        // Wide-row fast 2-pass path: non-multiple-of-4 rows with ties + LLM vocab size.
        {
            const int rows = 3, cols = 65537;
            auto wide = _Input({rows, cols}, NCHW);
            auto ptr = wide->writeMap<float>();
            std::vector<int> expect = {0, 40961, cols - 1};
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    ptr[r * cols + c] = -10.0f;
                }
            }
            // Row 0: all equal -> first index (0) wins tie-break.
            // Row 1: duplicate max -> smaller index (40961) wins over 65000.
            ptr[1 * cols + 40961] = 12.5f;
            ptr[1 * cols + 65000] = 12.5f;
            // Row 2: max at tail index (cols - 1).
            ptr[2 * cols + (cols - 1)] = 7.25f;
            wide->unMap();
            auto out = _ArgMax(wide, -1);
            auto got = out->readMap<int>();
            if (!checkVector<int>(got, expect.data(), rows, 0)) {
                MNN_ERROR("ArgMaxTest wide non-div4 tie-break failed!\n");
                return false;
            }
        }
        {
            const int cols = 248320;
            auto vocab = _Input({1, 1, cols}, NCHW);
            auto ptr = vocab->writeMap<float>();
            for (int c = 0; c < cols; ++c) {
                ptr[c] = -3.0f + float(c % 17) * 0.125f;
            }
            const int expect = 197321;
            ptr[expect] = 19.5f;
            vocab->unMap();
            auto out = _ArgMax(vocab, -1);
            auto got = out->readMap<int>();
            if (!checkVector<int>(got, &expect, 1, 0)) {
                MNN_ERROR("ArgMaxTest 248320 vocab failed!\n");
                return false;
            }
        }
        return true;
    }
};
class ArgMinTest : public MNNTestCase {
public:
    virtual ~ArgMinTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({4, 4}, NHWC);
        // auto input = _Input({128 * 160, 4}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 2.0,   -3.0, 4.0,  5.0,  -6.0, 7.0,   -8.0,
                                  -9.0, -10.0, 11.0, 12.0, 13.0, 14.0, -15.0, -16.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 16 * sizeof(float));
        input->unMap();
        auto output_0                           = _ArgMin(input, 0);
        auto output_1                           = _ArgMin(input, 1);
        const std::vector<int> expectedOutput_0 = {2, 2, 3, 3};
        const std::vector<int> expectedOutput_1 = {2, 3, 1, 3};
        auto gotOutput_0                        = output_0->readMap<int>();
        auto gotOutput_1                        = output_1->readMap<int>();
        if (!checkVector<int>(gotOutput_0, expectedOutput_0.data(), 4, 0)) {
            MNN_ERROR("ArgMinTest test axis_0 failed!\n");
            return false;
        }
        if (!checkVector<int>(gotOutput_1, expectedOutput_1.data(), 4, 0)) {
            MNN_ERROR("ArgMinTest test axis_1 failed!\n");
            return false;
        }
        {
            const int rows = 2, cols = 65537;
            auto wide = _Input({rows, cols}, NCHW);
            auto ptr = wide->writeMap<float>();
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    ptr[r * cols + c] = 5.0f;
                }
            }
            ptr[0 * cols + 12345] = -8.5f;
            ptr[0 * cols + 54321] = -8.5f;
            ptr[1 * cols + (cols - 1)] = -11.0f;
            wide->unMap();
            const std::vector<int> expect = {12345, cols - 1};
            auto out = _ArgMin(wide, -1);
            auto got = out->readMap<int>();
            if (!checkVector<int>(got, expect.data(), rows, 0)) {
                MNN_ERROR("ArgMinTest wide non-div4 tie-break failed!\n");
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ArgMaxTest, "op/argmax");
MNNTestSuiteRegister(ArgMinTest, "op/argmin");
