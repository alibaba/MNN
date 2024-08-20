//
//  RasrerTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/12/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class RasrerTest : public MNNTestCase {
public:
    virtual ~RasrerTest() = default;
    bool _run(int precision, bool lazy) {
        auto input = _Input({2, 2}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1, 2, 3, 4};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        // transpose
        auto output                             = _Raster({input}, {0, 4, 1, 2, 0, 4, 2, 1, 1, 2, 2}, {2, 2});
        const std::vector<float> expectedOutput = {1, 3, 2, 4};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("RasterTest transpose test failed!\n");
            return false;
        }
        auto output0                             = _Raster({input}, {2, 4, 2, 1, 0, 4, 2, 1, 1, 1, 2}, {2});
        const std::vector<float> expectedOutput0 = {3, 4};
        auto gotOutput0                          = output0->readMap<float>();
        if (!checkVector<float>(gotOutput0, expectedOutput0.data(), 2, 0.01)) {
            MNN_ERROR("RasterTest slice test failed!\n");
            return false;
        }
        return true;
    }
    virtual bool run(int precision) {
        ExecutorScope::Current()->lazyEval = false;
        auto res = _run(precision, false);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->lazyEval = true;
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_CONTENT);
        res = _run(precision, true);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_FULL);
        res = _run(precision, true);
        return res;
    }

};
MNNTestSuiteRegister(RasrerTest, "op/raster");

class BlitC4Test : public MNNTestCase {
public:
    virtual ~BlitC4Test() = default;
    bool _run(int precision, bool lazy) {
        int w = 1;
        int h = 1;
        int n = 16;
        int c = 5;
        auto input0 = _Input({n, c, h, w}, NCHW);
        auto input1 = _Input({n, c, h, w}, NCHW);
        auto input2 = _Input({n, c, h, w}, NCHW);
        std::vector<float*> inputPtr = {
            input0->writeMap<float>(),
            input1->writeMap<float>(),
            input2->writeMap<float>(),
        };

        int p = (int)inputPtr.size();
        std::vector<float> outputData(n * c * h * w * p);
        float current = 0.0f;
        for (int pp=0; pp<p; ++pp) {
            auto ptr = inputPtr[pp];
            auto dstptr = outputData.data() + (p-pp-1) * n * c * h * w;
            for (int u=0; u<n; ++u) {
                auto ptrn = ptr + u * c * h * w;
                auto dstptrn = dstptr + u * c * h * w;
                for (int v=0; v<c; ++v) {
                    auto ptrv = ptrn + v * h * w;
                    auto dstptrv = dstptrn + v * h * w;
                    for (int y=0; y<h; ++y) {
                        for (int x=0; x<w; ++x) {
                            ptrv[y*w+x] = current;
                            dstptrv[y*w+x] = current;
                            current = current + 0.01f;
                        }
                    }
                }
            }
        }
        
        input0 = _Convert(input0, NC4HW4);
        input1 = _Convert(input1, NC4HW4);
        input2 = _Convert(input2, NC4HW4);
        auto output = _RasterRaw({input0, input1, input2}, {
            /**
             region.src.offset = _GET(0);
             region.src.stride[0] = _GET(1);
             region.src.stride[1] = _GET(2);
             region.src.stride[2] = _GET(3);
             region.dst.offset = _GET(4);
             region.dst.stride[0] = _GET(5);
             region.dst.stride[1] = _GET(6);
             region.dst.stride[2] = _GET(7);
             region.size[0] = _GET(8);
             region.size[1] = _GET(9);
             region.size[2] = _GET(10);
             region.origin = inputs[j];
             */
            0, w*h, 0, 0,   0, w*h, 0, 0, n * c, 1, 1,
            0, w*h, 0, 0,   n * c * w * h, w*h, 0, 0, n * c, 1, 1,
            0, w*h, 0, 0,   2 * n * c * w * h, w*h, 0, 0, n * c, 1, 1
        }, {p*n, c, h, w}, halide_type_of<float>(), NC4HW4);
        output = _Convert(output, NCHW);
        output = _Reshape(output, {p, -1});
        output = _Reverse(output, _Scalar<int>(0));
        auto outputPtr = output->readMap<float>();
        if (!checkVector<float>(outputPtr, outputData.data(), n * c * h * w * p, 0.01f)) {
            MNN_ERROR("blitc4 test failed!\n");
            return false;
        }
        return true;
    }
    virtual bool run(int precision) {
        ExecutorScope::Current()->lazyEval = false;
        auto res = _run(precision, false);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->lazyEval = true;
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_CONTENT);
        res = _run(precision, true);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_FULL);
        res = _run(precision, true);
        return res;
    }

};
MNNTestSuiteRegister(BlitC4Test, "op/blitc4");
