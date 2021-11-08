//
//  ReshapeTest.cpp
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
class ReshapeNCHWTest : public MNNTestCase {
public:
    virtual ~ReshapeNCHWTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({4}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        const int shape_data[]                  = {1, 4, 1, 1};
        auto shape                              = _Const(shape_data, {4}, NCHW, halide_type_of<int>());
        auto output                             = _Reshape(input, shape);
        const std::vector<float> expectedOutput = {-1.0, -2.0, 3.0, 4.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("ReshapeNCHWTest test failed!\n");
            return false;
        }
        auto gotDim = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), shape_data, 4, 0)) {
            MNN_ERROR("ReshapeNCHWTest test failed!\n");
            return false;
        }
        auto format = output->getInfo()->order;
        if (NCHW != format) {
            MNN_ERROR("ReshapeNCHWTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ReshapeNHWCTest : public MNNTestCase {
public:
    virtual ~ReshapeNHWCTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({4}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        const int shape_data[]                  = {1, 1, 1, 4};
        auto shape                              = _Const(shape_data, {4}, NCHW, halide_type_of<int>());
        auto output                             = _Reshape(input, shape);
        const std::vector<float> expectedOutput = {-1.0, -2.0, 3.0, 4.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("ReshapeNHWCTest test failed!\n");
            return false;
        }
        auto gotDim = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), shape_data, 4, 0)) {
            MNN_ERROR("ReshapeNHWCTest test failed!\n");
            return false;
        }
        auto format = output->getInfo()->order;
        if (NHWC != format) {
            MNN_ERROR("ReshapeNHWCTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ReshapeNC4HW4Test : public MNNTestCase {
public:
    virtual ~ReshapeNC4HW4Test() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 1, 1, 64}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                                  14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
                                  27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
                                  40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0,
                                  53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 64 * sizeof(float));
        input->unMap();
        input                                   = _Convert(input, NC4HW4);
        auto output                             = _Reshape(input, {1, 4, 4, 4}, NCHW);
        output                                  = _Convert(output, NCHW);
        const std::vector<float> expectedOutput = {
            1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
            49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 64, 0.01)) {
            MNN_ERROR("ReshapeNC4HW4Test test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim = {1, 4, 4, 4};
        auto gotDim                        = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
            MNN_ERROR("ReshapeNHWCTest test failed!\n");
            return false;
        }
        auto format = output->getInfo()->order;
        if (NCHW != format) {
            MNN_ERROR("ReshapeNC4HW4Test test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ReshapeNCHWTest, "op/reshape/nchw");
MNNTestSuiteRegister(ReshapeNHWCTest, "op/reshape/nhwc");
MNNTestSuiteRegister(ReshapeNC4HW4Test, "op/reshape/nc4hw4");
