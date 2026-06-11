//
//  UniqueTest.cpp
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
class UniqueTest : public MNNTestCase {
public:
    virtual ~UniqueTest() = default;
    virtual bool run(int precision) {
        const int xData[] = {1, 1, 2, 4, 4, 4, 7, 8, 8};
        auto makeUnique = [&xData](int outputCount) {
            auto input = _Input({9}, NHWC, halide_type_of<int>());
            input->setName("input_x");
            auto xPtr = input->writeMap<int>();
            memcpy(xPtr, xData, 9 * sizeof(int));
            input->unMap();
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type = MNN::OpType_Unique;
            op->main.type = MNN::OpParameter_NONE;
            op->main.value = nullptr;
            return Expr::create(std::move(op), {input}, outputCount);
        };
        auto checkDim = [](VARP output, int expected, const char* name) {
            auto info = output->getInfo();
            if (nullptr == info || info->dim.size() != 1 || info->dim[0] != expected) {
                MNN_ERROR("UniqueTest %s shape test failed!\n", name);
                return false;
            }
            return true;
        };
        {
            auto expr = makeUnique(2);
            auto output0 = Variable::create(expr, 0);
            auto output1 = Variable::create(expr, 1);
            const std::vector<int> expectedOutput0 = {1, 2, 4, 7, 8};
            const std::vector<int> expectedOutput1 = {0, 0, 1, 2, 2, 2, 3, 4, 4};
            if (!checkDim(output0, 5, "unique") || !checkDim(output1, 9, "idx")) {
                return false;
            }
            if (!checkVector<int>(output0->readMap<int>(), expectedOutput0.data(), 5, 0) ||
                !checkVector<int>(output1->readMap<int>(), expectedOutput1.data(), 9, 0)) {
                MNN_ERROR("UniqueTest two-output test failed!\n");
                return false;
            }
        }
        {
            auto expr = makeUnique(4);
            auto output0 = Variable::create(expr, 0);
            auto output1 = Variable::create(expr, 1);
            auto output2 = Variable::create(expr, 2);
            auto output3 = Variable::create(expr, 3);
            const std::vector<int> expectedOutput0 = {1, 2, 4, 7, 8};
            const std::vector<int> expectedOutput1 = {0, 2, 3, 6, 7};
            const std::vector<int> expectedOutput2 = {0, 0, 1, 2, 2, 2, 3, 4, 4};
            const std::vector<int> expectedOutput3 = {2, 1, 3, 1, 2};
            if (!checkDim(output0, 5, "unique") || !checkDim(output1, 5, "indices") ||
                !checkDim(output2, 9, "inverse") || !checkDim(output3, 5, "counts")) {
                return false;
            }
            if (!checkVector<int>(output0->readMap<int>(), expectedOutput0.data(), 5, 0) ||
                !checkVector<int>(output1->readMap<int>(), expectedOutput1.data(), 5, 0) ||
                !checkVector<int>(output2->readMap<int>(), expectedOutput2.data(), 9, 0) ||
                !checkVector<int>(output3->readMap<int>(), expectedOutput3.data(), 5, 0)) {
                MNN_ERROR("UniqueTest four-output test failed!\n");
                return false;
            }
        }
        {
            auto expr = makeUnique(3);
            auto output0 = Variable::create(expr, 0);
            auto output1 = Variable::create(expr, 1);
            auto output2 = Variable::create(expr, 2);
            const std::vector<int> expectedOutput0 = {1, 2, 4, 7, 8};
            const std::vector<int> expectedOutput1 = {0, 2, 3, 6, 7};
            const std::vector<int> expectedOutput2 = {0, 0, 1, 2, 2, 2, 3, 4, 4};
            if (!checkDim(output0, 5, "unique") || !checkDim(output1, 5, "indices") ||
                !checkDim(output2, 9, "inverse")) {
                return false;
            }
            if (!checkVector<int>(output0->readMap<int>(), expectedOutput0.data(), 5, 0) ||
                !checkVector<int>(output1->readMap<int>(), expectedOutput1.data(), 5, 0) ||
                !checkVector<int>(output2->readMap<int>(), expectedOutput2.data(), 9, 0)) {
                MNN_ERROR("UniqueTest three-output test failed!\n");
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(UniqueTest, "op/unique");
