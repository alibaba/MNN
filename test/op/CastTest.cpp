//
//  CastTest.cpp
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

static bool int32ToInt8() {
    auto input = _Input({4, 1, 1, 3}, NHWC, halide_type_of<int32_t>());
    input->setName("input_tensor");
    // set input data
    const int inpudata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto inputPtr          = input->writeMap<int32_t>();
    memcpy(inputPtr, inpudata, 12 * sizeof(int32_t));
    input->unMap();
    auto output                              = _Cast<int8_t>(input);
    const std::vector<int8_t> expectedOutput = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    {
        auto gotOutput = output->readMap<int8_t>();
        if (!checkVector<int8_t>(gotOutput, expectedOutput.data(), 12, 0)) {
            FUNC_PRINT(1);
            for (int i = 0; i < 12; ++i) {
                MNN_PRINT("Correct: %d - Compute: %d\n", expectedOutput[i], gotOutput[i]);
            }
            return false;
        }
    }
    output = _Cast<int32_t>(output);
    {
        auto gotOutput = output->readMap<int32_t>();
        if (!checkVector<int32_t>(gotOutput, inpudata, 12, 0)) {
            FUNC_PRINT(1);
            for (int i = 0; i < 12; ++i) {
                MNN_PRINT("Correct: %d - Compute: %d\n", inpudata[i], gotOutput[i]);
            }
            return false;
        }
    }
    return true;
}
static bool int32ToFloat32() {
    auto input = _Input({4, 1, 1, 3}, NHWC);
    input->setName("input_tensor");
    // set input data
    const float inpudata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    auto inputPtr          = input->writeMap<float>();
    memcpy(inputPtr, inpudata, 12 * sizeof(float));
    input->unMap();
    auto output                              = _Cast<int32_t>(input);
    const std::vector<int32_t> expectedOutput = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    {
        auto gotOutput = output->readMap<int32_t>();
        if (!checkVector<int32_t>(gotOutput, expectedOutput.data(), 12, 0)) {
            FUNC_PRINT(1);
            for (int i = 0; i < 12; ++i) {
                MNN_PRINT("Correct: %d - Compute: %d\n", expectedOutput[i], gotOutput[i]);
            }
            return false;
        }
    }
    output = _Cast<float>(output);
    {
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, inpudata, 12, 0.01)) {
            FUNC_PRINT(1);
            for (int i = 0; i < 12; ++i) {
                MNN_PRINT("Correct: %f - Compute: %f\n", inpudata[i], gotOutput[i]);
            }
            return false;
        }
    }
    return true;
}
static bool int8ToFloat32() {
    auto input = _Input({4, 1, 1, 3}, NHWC);
    input->setName("input_tensor");
    // set input data
    const float inpudata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    auto inputPtr          = input->writeMap<float>();
    memcpy(inputPtr, inpudata, 12 * sizeof(float));
    input->unMap();
    auto output                              = _Cast<int8_t>(input);
    const std::vector<int8_t> expectedOutput = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    {
        auto gotOutput = output->readMap<int8_t>();
        if (!checkVector<int8_t>(gotOutput, expectedOutput.data(), 12, 0)) {
            FUNC_PRINT(1);
            for (int i = 0; i < 12; ++i) {
                MNN_PRINT("Correct: %d - Compute: %d\n", expectedOutput[i], gotOutput[i]);
            }
            return false;
        }
    }
    output = _Cast<float>(output);
    {
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, inpudata, 12, 0.01)) {
            FUNC_PRINT(1);
            for (int i = 0; i < 12; ++i) {
                MNN_PRINT("Correct: %f - Compute: %f\n", inpudata[i], gotOutput[i]);
            }
            return false;
        }
    }
    return true;
}

static bool uint8ToFloat32() {
    auto input = _Input({4, 1, 1, 3}, NHWC);
    input->setName("input_tensor");
    // set input data
    const float inpudata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    auto inputPtr          = input->writeMap<float>();
    memcpy(inputPtr, inpudata, 12 * sizeof(float));
    input->unMap();
    auto output                              = _Cast<uint8_t>(input);
    const std::vector<uint8_t> expectedOutput = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    {
        auto gotOutput = output->readMap<uint8_t>();
        if (!checkVector<uint8_t>(gotOutput, expectedOutput.data(), 12, 0)) {
            FUNC_PRINT(1);
            for (int i = 0; i < 12; ++i) {
                MNN_PRINT("Correct: %d - Compute: %d\n", expectedOutput[i], gotOutput[i]);
            }
            return false;
        }
    }
    output = _Cast<float>(output);
    {
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, inpudata, 12, 0.01)) {
            FUNC_PRINT(1);
            for (int i = 0; i < 12; ++i) {
                MNN_PRINT("Correct: %f - Compute: %f\n", inpudata[i], gotOutput[i]);
            }
            return false;
        }
    }
    return true;
}

class CastTest : public MNNTestCase {
public:
    virtual ~CastTest() = default;
    virtual bool run(int precision) {
        auto res = int8ToFloat32();
        res = res && int32ToFloat32();
        res = res && int32ToInt8();
        res = res && uint8ToFloat32();
        return res;
    }
};
MNNTestSuiteRegister(CastTest, "op/cast");
