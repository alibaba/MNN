//
//  ConcatTest.cpp
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


class ConcatTest : public MNNTestCase {
public:
    virtual ~ConcatTest() = default;
    virtual bool run(int precision) {

        return commonCase() &&
               zeroCase() &&
               allZeroCase();
    }
    bool commonCase() {
    auto input1 = _Input({2, 2}, NCHW);
    input1->setName("input1");
    // set input data
    const float input1_data[] = {1.0, 2.0, 3.0, 4.0};
    auto input1Ptr            = input1->writeMap<float>();
    memcpy(input1Ptr, input1_data, 4 * sizeof(float));
    input1->unMap();
    auto input2 = _Input({2, 2}, NCHW);
    input2->setName("input2");
    // set input data
    const float input2_data[] = {5.0, 6.0, 7.0, 8.0};
    auto input2Ptr            = input2->writeMap<float>();
    memcpy(input2Ptr, input2_data, 4 * sizeof(float));
    input2->unMap();
    auto output                             = _Concat({input1, input2}, 1);
    const std::vector<float> expectedOutput = {1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0};
    auto gotOutput                          = output->readMap<float>();
    if (!checkVector<float>(gotOutput, expectedOutput.data(), 8, 0.0001)) {
        MNN_ERROR("ConcatTest test failed!\n");
        return false;
    }
    const std::vector<int> expectedDim = {2, 4};
    auto gotDim                        = output->getInfo()->dim;
    if (!checkVector<int>(gotDim.data(), expectedDim.data(), 2, 0)) {
        MNN_ERROR("ConcatTest test failed!\n");
        return false;
    }
    return true;
}


    bool zeroCase() {

        auto input1 = _Input({2, 2}, NCHW);
        input1->setName("input1");
        // set input data
        const float input1_data[] = {1.0, 2.0, 3.0, 4.0};
        auto input1Ptr            = input1->writeMap<float>();
        memcpy(input1Ptr, input1_data, 4 * sizeof(float));
        input1->unMap();
        auto input2 = _Input({2, 0}, NCHW);
        input2->setName("input2");
        // set input data
        auto input2Ptr            = input2->writeMap<float>();
        input2->unMap();
        auto output                             = _Concat({input1, input2}, 1);
        const std::vector<float> expectedOutput = {1.0, 2.0, 3.0, 4.0,};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.0001)) {
            MNN_ERROR("ConcatTest zero case failed!\n");
            return false;
        }
        const std::vector<int> expectedDim = {2, 2};
        auto gotDim                        = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), expectedDim.data(), 2, 0)) {
            MNN_ERROR("ConcatTest zero case failed!\n");
            return false;
        }

        return true;
    }

    bool allZeroCase() {

        auto input1 = _Input({2, 0}, NCHW);
        input1->setName("input1");
        input1->unMap();
        auto input2 = _Input({2, 0}, NCHW);
        input2->setName("input2");
        input2->unMap();
        auto output                             = _Concat({input1, input2}, 1);

        auto gotOutput                          = output->readMap<float>();
        const std::vector<int> expectedDim = {2, 0};
        auto gotDim                        = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), expectedDim.data(), 2, 0)) {
            MNN_ERROR("ConcatTest all zero case failed!\n");
            return false;
        }

        return true;
    }

};
MNNTestSuiteRegister(ConcatTest, "op/concat");
