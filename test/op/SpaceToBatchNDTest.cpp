//
//  SpaceToBatchNDTest.cpp
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
class SpaceToBatchNDTest : public MNNTestCase {
public:
    virtual ~SpaceToBatchNDTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({3, 1, 2, 2}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 12 * sizeof(float));
        input->unMap();
        const int blockshapedata[]              = {2, 2};
        const int paddingdata[]                 = {0, 0, 0, 0};
        auto block_shape                        = _Const(blockshapedata,
                                  {
                                      2,
                                  },
                                  NCHW, halide_type_of<int>());
        auto paddings                           = _Const(paddingdata, {2, 2}, NCHW, halide_type_of<int>());
        input                                   = _Convert(input, NC4HW4);
        // 1 input and 2 params
        auto tmp                                = _SpaceToBatchND(input, block_shape, paddings);
        auto output                             = _Convert(tmp, NCHW);
        // 3 inputs and 0 param
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->type       = MNN::OpType_SpaceToBatchND;
        auto _tmp = Variable::create(Expr::create(std::move(op), {input, block_shape, paddings}));
        auto _output                             = _Convert(_tmp, NCHW);
        auto checkOutput = [](VARP output) {
            const std::vector<float> expectedOutput = {1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 12, 0.01)) {
                MNN_ERROR("SpaceToBatchNDTest test failed!\n");
                return false;
            }
            const std::vector<int> expectedDims = {12, 1, 1, 1};
            auto gotDims                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDims.data(), expectedDims.data(), 4, 0)) {
                MNN_ERROR("SpaceToBatchNDTest test failed!\n");
                return false;
            }
            return true;
        };
        return checkOutput(output) && checkOutput(_output);
    }
};
MNNTestSuiteRegister(SpaceToBatchNDTest, "op/space_to_batch_nd");
