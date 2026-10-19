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

class SpaceBatchPaddingTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const float original[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        const float padded[2][18] = {
            {1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12, 0, 0, 0, 0, 0, 0},
            {1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12, 0, 0, 0, 0, 0, 0}};
        for (int axis = 0; axis < 2; ++axis) {
            const int blocks[] = {axis == 0 ? 3 : 1, axis == 0 ? 1 : 3};
            const int padding[] = {0, axis == 0 ? 1 : 0, 0, axis == 0 ? 0 : 1};
            const std::vector<int> paddedShape = axis == 0 ? std::vector<int>{3, 3, 1, 2} : std::vector<int>{3, 3, 2, 1};
            auto block = _Const(blocks, {2}, NCHW, halide_type_of<int>());
            auto pad = _Const(padding, {2, 2}, NCHW, halide_type_of<int>());
            for (int inverse = 0; inverse < 2; ++inverse) {
                const std::vector<int> shape = inverse ? paddedShape : std::vector<int>{1, 3, 2, 2};
                const std::vector<int> expectedShape = inverse ? std::vector<int>{1, 3, 2, 2} : paddedShape;
                auto input = _Input(shape, NCHW);
                ::memcpy(input->writeMap<float>(), inverse ? padded[axis] : original,
                         inverse ? sizeof(padded[axis]) : sizeof(original));
                input = _Convert(input, NC4HW4);
                auto output = inverse ? _BatchToSpaceND(input, block, pad) : _SpaceToBatchND(input, block, pad);
                output = _Convert(output, NCHW);
                auto result = output->readMap<float>();
                if (result == nullptr || output->getInfo()->dim != expectedShape) {
                    MNN_ERROR("Space/batch transform with a fully padded block failed\n");
                    return false;
                }
                for (int i = 0; i < (inverse ? 12 : 18); ++i) {
                    if (result[i] != (inverse ? original[i] : padded[axis][i])) {
                        MNN_ERROR("Space/batch padding mismatch at %d\n", i);
                        return false;
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(SpaceBatchPaddingTest, "op/space_batch_padding");
