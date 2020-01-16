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
    virtual bool run() {
        auto input = _Input({4,1,1,3}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 12 * sizeof(float));
        input->unMap();
        const int blockshapedata[] = {2,2};
        const int cropsdata[] = {0,0,0,0};
        auto block_shape = _Const(blockshapedata,{2,},NCHW,halide_type_of<int>());
        auto crops = _Const(cropsdata,{2,2},NCHW,halide_type_of<int>());
        input = _Convert(input, NC4HW4);
        auto tmp = _BatchToSpaceND(input, block_shape, crops);
	auto output = _Convert(tmp, NHWC);
        const std::vector<float> expectedOutput = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 12, 0.01)) {
            MNN_ERROR("BatchToSpaceNDTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDims = {1,2,2,3};
        auto  gotDims = output->getInfo()->dim;
  	if (!checkVector<int>(gotDims.data(), expectedDims.data(), 4, 0)) {
            MNN_ERROR("BatchToSpaceNDTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(BatchToSpaceNDTest, "op/batch_to_space_nd");
