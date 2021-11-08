//
//  TransposeTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class TransposeTest : public MNNTestCase {
public:
    virtual ~TransposeTest() = default;
    virtual bool run(int precision) {
        const int n = 2, c = 3, h = 4, w = 4;
        const std::vector<float> inputData      = {0.5488,
                                              0.7152,
                                              0.6028,
                                              0.5449,
                                              0.4237,
                                              0.6459,
                                              0.4376,
                                              0.8918,
                                              0.9637,
                                              0.3834,
                                              0.7917,
                                              0.5289,
                                              0.568,
                                              0.9256,
                                              0.071,
                                              0.0871,

                                              0.0202,
                                              0.8326,
                                              0.7782,
                                              0.87,
                                              0.9786,
                                              0.7992,
                                              0.4615,
                                              0.7805,
                                              0.1183,
                                              0.6399,
                                              0.1434,
                                              0.9447,
                                              0.5218,
                                              0.4147,
                                              0.2646,
                                              0.7742,

                                              0.4562,
                                              0.5684,
                                              0.0188,
                                              0.6176,
                                              0.6121,
                                              0.6169,
                                              0.9437,
                                              0.6818,
                                              0.3595,
                                              0.437,
                                              0.6976,
                                              0.0602,
                                              0.6668,
                                              0.6706,
                                              0.2104,
                                              0.1289

                                              ,
                                              0.5488,
                                              0.7152,
                                              0.6028,
                                              0.5449,
                                              0.4237,
                                              0.6459,
                                              0.4376,
                                              0.8918,
                                              0.9637,
                                              0.3834,
                                              0.7917,
                                              0.5289,
                                              0.568,
                                              0.9256,
                                              0.071,
                                              0.0871,

                                              0.0202,
                                              0.8326,
                                              0.7782,
                                              0.87,
                                              0.9786,
                                              0.7992,
                                              0.4615,
                                              0.7805,
                                              0.1183,
                                              0.6399,
                                              0.1434,
                                              0.9447,
                                              0.5218,
                                              0.4147,
                                              0.2646,
                                              0.7742,

                                              0.4562,
                                              0.5684,
                                              0.0188,
                                              0.6176,
                                              0.6121,
                                              0.6169,
                                              0.9437,
                                              0.6818,
                                              0.3595,
                                              0.437,
                                              0.6976,
                                              0.0602,
                                              0.6668,
                                              0.6706,
                                              0.2104,
                                              0.1289};
        const std::vector<float> expectedOutput = {
            0.5488, 0.0202, 0.4562, 0.4237, 0.9786, 0.6121, 0.9637, 0.1183, 0.3595, 0.5680, 0.5218, 0.6668,

            0.7152, 0.8326, 0.5684, 0.6459, 0.7992, 0.6169, 0.3834, 0.6399, 0.4370, 0.9256, 0.4147, 0.6706,

            0.6028, 0.7782, 0.0188, 0.4376, 0.4615, 0.9437, 0.7917, 0.1434, 0.6976, 0.0710, 0.2646, 0.2104,

            0.5449, 0.8700, 0.6176, 0.8918, 0.7805, 0.6818, 0.5289, 0.9447, 0.0602, 0.0871, 0.7742, 0.1289,

            0.5488, 0.0202, 0.4562, 0.4237, 0.9786, 0.6121, 0.9637, 0.1183, 0.3595, 0.5680, 0.5218, 0.6668,

            0.7152, 0.8326, 0.5684, 0.6459, 0.7992, 0.6169, 0.3834, 0.6399, 0.4370, 0.9256, 0.4147, 0.6706,

            0.6028, 0.7782, 0.0188, 0.4376, 0.4615, 0.9437, 0.7917, 0.1434, 0.6976, 0.0710, 0.2646, 0.2104,

            0.5449, 0.8700, 0.6176, 0.8918, 0.7805, 0.6818, 0.5289, 0.9447, 0.0602, 0.0871, 0.7742, 0.1289};
        auto input = _Input({n, c, h, w}, NCHW, halide_type_of<float>());
        input->setName("input_tensor");
        auto inputPtr = input->writeMap<float>();
        memcpy(inputPtr, inputData.data(), inputData.size() * sizeof(float));
        input->unMap();
        auto output    = _Transpose(input, {0, 3, 2, 1});
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 5, 0.01)) {
            MNN_ERROR("TransposeTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(TransposeTest, "op/transpose");
