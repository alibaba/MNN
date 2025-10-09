//
//  WhereTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/11/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
static void print(VARP xc, std::string name) {
    MNN_PRINT("%s begin:\n", name.c_str());
    xc->readMap<void>();
    xc->getTensor()->print();
    MNN_PRINT("%s end:\n", name.c_str());
}
static bool _testFromIssue() {
    {
        // 初始化
        VARP var = MNN::Express::_Input({10}, Dimensionformat::NCHW, halide_type_of<float>());
        float *pData = var->writeMap<float>();
        std::vector<float> data = {
            0.5f, 0.5f, 1.0f, 1.0f, 1.5f, 1.5f, 1.0f, 1.0f, 0.2f, 0.3f,
        };
        memcpy(pData, data.data(), var->getInfo()->size*sizeof(float));
        
        auto xc = _Greater(var, _Scalar<float>(1.0));
        print(xc, "xc");
        
        auto w = _Where(xc);
        print(w, "where");
        if (w->getInfo()->size != 2) {
            return false;
        }
        if (w->readMap<int>()[0] != 4 || w->readMap<int>()[1] != 5) {
            return false;
        }
    }
    
    {
        VARP var = _Input({10}, Dimensionformat::NCHW, halide_type_of<float>());
        auto pData = var->writeMap<float>();
        std::vector<float> data = {
            0.5f, 0.5f, 1.0f, 1.0f, 1.5f, 1.5f, 1.0f, 1.0f, 0.2f, 0.3f,
        };
        memcpy(pData, data.data(), var->getInfo()->size * sizeof(float));
        auto index = _Sort(var, 0, true, true);
        print(index, "Sort index");
        auto value = _Sort(var, 0, false, true);
        print(value, "Sort value");
    }
    return true;
}

class WhereTest : public MNNTestCase {
public:
    virtual ~WhereTest() = default;
    virtual bool run(int precision) {
        auto res = _testFromIssue();
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        return commonCase() &&
               zeroCase();

    }
    bool commonCase() {
        auto input = _Input({2, 3}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = { 1.0, 0.0, 2.0,
                                   3.0, 0.0, 4.0 };
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 6 * sizeof(float));
        input->unMap();
        auto output                           = _Where(input);
        const std::vector<int> expectedOutput = {0, 0, 0, 2, 1, 0, 1, 2};
        const std::vector<int> expectedShape = {4, 2};

        auto realShape = output->getInfo()->dim;
        if (!checkVector<int>(realShape.data(), expectedShape.data(), 2, 0)) {
            MNN_ERROR("WhereTest shape mismatch!\n");
            return false;
        }

        auto gotOutput                        = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("WhereTest test failed!\n");
            return false;
        }
        return true;
    }

    bool zeroCase() {

        auto input = _Input({2, 4}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata_zero[] = { 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0 };
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata_zero, 8 * sizeof(float));
        input->unMap();
        auto output                           = _Where(input);
        const std::vector<int> expectedOutput = {};
        const std::vector<int> expectedShape = {0, 2};

        auto realShape = output->getInfo()->dim;
        if (!checkVector<int>(realShape.data(), expectedShape.data(), 2, 0)) {
            MNN_ERROR("WhereTest zero shape mismatch!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(WhereTest, "op/where");
