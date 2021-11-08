//
//  BatchToSpaceNDTest.cpp
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
class BatchToSpaceNDTest : public MNNTestCase {
public:
    virtual ~BatchToSpaceNDTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({4, 1, 1, 3}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 12 * sizeof(float));
        input->unMap();
        const int blockshapedata[] = {2, 2};
        const int cropsdata[]      = {0, 0, 0, 0};
        auto block_shape           = _Const(blockshapedata,
                                  {
                                      2,
                                  },
                                  NCHW, halide_type_of<int>());
        auto crops                 = _Const(cropsdata, {2, 2}, NCHW, halide_type_of<int>());
        input                      = _Convert(input, NC4HW4);
        if (false) {
            auto inputPtr2                          = input->readMap<float>();
            const std::vector<float> expectedOutput = {1.0f, 2.0f, 3.0f, 0.0f, 4.0f,  5.0f,  6.0f,  0.0f,
                                                       7.0f, 8.0f, 9.0f, 0.0f, 10.0f, 11.0f, 12.0f, 0.0f};
            if (!checkVector<float>(inputPtr2, expectedOutput.data(), expectedOutput.size(), 0.01)) {
                MNN_ERROR("BatchToSpaceNDTest test failed!\n");
                for (int i = 0; i < expectedOutput.size(); ++i) {
                    MNN_PRINT("Correct: %f - Compute: %f\n", expectedOutput[i], inputPtr2[i]);
                }
                return false;
            }
        }
        // 1 input and 2 params
        auto tmp                                = _BatchToSpaceND(input, block_shape, crops);
        auto output                             = _Convert(tmp, NHWC);
        // 3 inputs and 1 param
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->type = MNN::OpType_BatchToSpaceND;
        auto _tmp = Variable::create(Expr::create(std::move(op), {input, block_shape, crops}));
        auto _output = _Convert(tmp, NHWC);
        auto checkOutput = [](VARP output) {
            const std::vector<float> expectedOutput = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 12, 0.01)) {
                MNN_ERROR("BatchToSpaceNDTest test failed!\n");
                for (int i = 0; i < 12; ++i) {
                    MNN_PRINT("Correct: %f - Compute: %f\n", expectedOutput[i], gotOutput[i]);
                }
                return false;
            }
            const std::vector<int> expectedDims = {1, 2, 2, 3};
            auto gotDims                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDims.data(), expectedDims.data(), 4, 0)) {
                MNN_ERROR("BatchToSpaceNDTest test failed!\n");
                return false;
            }
            return true;
        };
        return checkOutput(output) && checkOutput(_output);
    }
};
MNNTestSuiteRegister(BatchToSpaceNDTest, "op/batch_to_space_nd");
