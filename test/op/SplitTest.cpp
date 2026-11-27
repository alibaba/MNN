//
//  SplitTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"

using namespace MNN::Express;
class SplitTest : public MNNTestCase {
public:
    virtual ~SplitTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({2, 4}, NCHW);
        input->setName("input");
        // set input data
        const float input_data[] = {1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0};
        auto inputPtr            = input->writeMap<float>();
        memcpy(inputPtr, input_data, 8 * sizeof(float));
        auto outputs                             = _Split(input, {1, 3}, 1);
        const std::vector<float> expectedOutput0 = {1.0, 3.0};
        auto gotOutput0                          = outputs[0]->readMap<float>();
        if (!checkVector<float>(gotOutput0, expectedOutput0.data(), 2, 0.0001)) {
            MNN_ERROR("SplitTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim0 = {2, 1};
        auto gotDim0                        = outputs[0]->getInfo()->dim;
        if (!checkVector<int>(gotDim0.data(), expectedDim0.data(), 2, 0)) {
            MNN_ERROR("SplitTest test failed!\n");
            return false;
        }
        const std::vector<float> expectedOutput1 = {2.0, 5.0, 6.0, 4.0, 7.0, 8.0};
        auto gotOutput1                          = outputs[1]->readMap<float>();
        if (!checkVector<float>(gotOutput1, expectedOutput1.data(), 6, 0.0001)) {
            MNN_ERROR("SplitTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim1 = {2, 3};
        auto gotDim1                        = outputs[1]->getInfo()->dim;
        if (!checkVector<int>(gotDim1.data(), expectedDim1.data(), 2, 0)) {
            MNN_ERROR("SplitTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(SplitTest, "op/split");

// Torch-style Slice (also emitted by the ONNX Split converter) on an axis that does not divide evenly.
// The last chunk must take the remainder instead of reading past the input.
class SplitUnevenTest : public MNNTestCase {
public:
    virtual ~SplitUnevenTest() = default;
    static bool check(const int* splitSize, int outputNumber, const std::vector<int>& expectLengths) {
        const int length = 10, inner = 2;
        std::vector<float> data(length * inner);
        for (int i = 0; i < (int)data.size(); ++i) {
            data[i] = (float)i;
        }
        auto input = _Const(data.data(), {length, inner}, NCHW);
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->type = MNN::OpType_Slice;
        op->main.type = MNN::OpParameter_Slice;
        auto param = new MNN::SliceT;
        param->axis = 0;
        param->sourceType = MNN::NetSource_TORCH;
        if (nullptr != splitSize) {
            param->slicePoints = {*splitSize};
        }
        op->main.value = param;
        auto expr = Expr::create(op.get(), {input}, outputNumber);
        int offset = 0;
        for (int i = 0; i < outputNumber; ++i) {
            auto output = Variable::create(expr, i);
            auto info = output->getInfo();
            if (nullptr == info || info->dim.size() != 2 || info->dim[0] != expectLengths[i] || info->dim[1] != inner) {
                MNN_ERROR("SplitUnevenTest: output %d has wrong shape, expect length %d\n", i, expectLengths[i]);
                return false;
            }
            if (!checkVector<float>(output->readMap<float>(), data.data() + offset, expectLengths[i] * inner, 0)) {
                MNN_ERROR("SplitUnevenTest: output %d has wrong values\n", i);
                return false;
            }
            offset += expectLengths[i] * inner;
        }
        return true;
    }
    virtual bool run(int precision) {
        // torch.split(x, 3) on length 10.
        const int splitSize = 3;
        if (!check(&splitSize, 4, {3, 3, 3, 1})) {
            return false;
        }
        // ONNX Split with num_outputs = 3 and no split sizes on length 10.
        return check(nullptr, 3, {4, 4, 2});
    }
};
MNNTestSuiteRegister(SplitUnevenTest, "op/split_uneven");

class SliceTest : public MNNTestCase {
public:
    virtual ~SliceTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({2, 9, 4}, NCHW, halide_type_of<int>());
        input->setName("input");
        // set input data
        auto size = input->getInfo()->size;
        auto iptr = input->writeMap<int>();
        for (int i=0; i<size; ++i) {
            int ci = i % 4;
            int co = i / 36;
            int area = (i / 4) % 9;
            iptr[i] = (ci+co*4) * 10 + area;
        }
        input->unMap();
        auto inputTran = _Reshape(_Transpose(input, {0, 2, 1}), {8, 9}, NCHW);
        std::vector<int> startDims = {1, 0};
        std::vector<int> sizeDims = {4, 9};
        auto start = _Const(startDims.data(), {2}, NCHW, halide_type_of<int>());
        auto sizeVar = _Const(sizeDims.data(), {2}, NCHW, halide_type_of<int>());

        auto output = _Slice(inputTran, start, sizeVar);
        auto oinfo = output->getInfo();
        if (oinfo->dim.size() != 2) {
            FUNC_PRINT(1);
            return false;
        }
        if (oinfo->dim[1] != 9 || oinfo->dim[0] != 4) {
            FUNC_PRINT(1);
            return false;
        }
        auto optr = output->readMap<int>();
        for (int i=0; i<4; ++i) {
            for (int j=0; j<9; ++j) {
                int expect = (i+1)*10+j;
                int compute = optr[i*9+j];
                if (expect != compute) {
                    MNN_ERROR("Error for i=%d - j=%d, %d:%d\n", i, j, expect, compute);
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(SliceTest, "op/slice");
