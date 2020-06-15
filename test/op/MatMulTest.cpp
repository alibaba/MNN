//
//  MatMulTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <utility>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "TestUtils.h"

#define TEST_RANDOM_SEED 100

using std::vector;
// C = A * B
static void reference_matmul(const vector<float>& matrix_a, const vector<float>& matrix_b, vector<float>& matrix_c, int width_a, int width_b, bool tranpose_a, bool tranpose_b) {
    int height_c = matrix_a.size() / width_a, width_c = width_b, length = width_a;
    int stride_a_h = width_a, stride_a_w = 1, stride_b_h = width_b, stride_b_w = 1;
    if (tranpose_a) {
        length = matrix_a.size() / width_a;
        stride_a_w = height_c = width_a;
        stride_a_h = 1;
    }
    if (tranpose_b) {
        width_c = matrix_b.size() / width_b;
        length = stride_b_w = width_b;
        stride_b_h = 1;
    }
    matrix_c.resize(height_c * width_c);
    for (int h = 0; h < height_c; ++h) {
        for (int w = 0; w < width_c; ++w) {
            float result = 0;
            for (int i = 0; i < length; ++i) {
                result += matrix_a[h * stride_a_h + i * stride_a_w] * matrix_b[i * stride_b_h + w * stride_b_w];
            }
            matrix_c[h * width_c + w] = result;
        }
    }
}

using namespace MNN::Express;
class MatMulCommonTest : public MNNTestCase {
public:
    virtual ~MatMulCommonTest() = default;
protected:
    static bool test(MNNForwardType type, const std::string& device_name, const std::string& test_op_name,
                     int height_a, int width_a, int height_b, int width_b, bool tranpose_a, bool tranpose_b) {
        auto input_a = _Input({height_a, width_a}, NCHW);
        auto input_b = _Input({height_b, width_b}, NCHW);
        auto output = _MatMul(input_a, input_b, tranpose_a, tranpose_b);
        vector<float> data_a, data_b, data_c;
        for (int i = 0; i < height_a * width_a; ++i) {
            data_a.push_back(rand() % 255 / 255.f);
        }
        for (int i = 0; i < height_b * width_b; ++i) {
            data_b.push_back(rand() % 255 / 255.f);
        }
        reference_matmul(data_a, data_b, data_c, width_a, width_b, tranpose_a, tranpose_b);
        ::memcpy(input_a->writeMap<float>(), data_a.data(), data_a.size() * sizeof(float));
        ::memcpy(input_b->writeMap<float>(), data_b.data(), data_b.size() * sizeof(float));
        if (!checkVectorByRelativeError<float>(output->readMap<float>(), data_c.data(), data_c.size(), 0.005)) {
            MNN_ERROR("%s: %d, %d, %d, %d, %d, %d test failed!\n", test_op_name.c_str(), height_a, width_a, height_b, width_b, tranpose_a, tranpose_b);
            return false;
        }
        return true;
    }
};

class MatMulTest : public MatMulCommonTest {
public:
    virtual ~MatMulTest() = default;
protected:
    static bool test(MNNForwardType type, const std::string& device_name) {
        srand(TEST_RANDOM_SEED);
        for (int height_c = 1; height_c <= 20; ++height_c) {
            for (int width_c = 1; width_c <= 20; ++width_c) {
                for (int length = 1; length <= 20; ++length) {
                    int height_a = height_c, height_b = length, width_a = length, width_b = width_c;
                    for (int tranpose_a = 0; tranpose_a <= 1; ++tranpose_a) {
                        int height_a = height_c, width_a = length;
                        if (tranpose_a == 1) {
                            std::swap(height_a, width_a);
                        }
                        for (int tranpose_b = 0; tranpose_b <= 1; ++tranpose_b) {
                            int height_b = length, width_b = width_c;
                            if (tranpose_b == 1) {
                                std::swap(height_b, width_b);
                            }
                            bool succ = MatMulCommonTest::test(type, device_name, "MatMul", height_a, width_a, height_b, width_b, tranpose_a != 0, tranpose_b != 0);
                            if (!succ) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};

class MatMulTestOnCPU : public MatMulTest {
public:
    virtual ~MatMulTestOnCPU() = default;
    virtual bool run() {
        return MatMulTest::test(MNN_FORWARD_CPU, "CPU");
    }
};

MNNTestSuiteRegister(MatMulTestOnCPU, "op/matmul");
