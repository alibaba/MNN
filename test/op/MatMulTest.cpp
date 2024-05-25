//
//  MatMulTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <utility>
#include <vector>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"

#define TEST_RANDOM_SEED 100

using std::vector;
// C = A * B
static void reference_matmul(const vector<float>& matrix_a, const vector<float>& matrix_b, vector<float>& matrix_c,
                             int width_a, int width_b, bool tranpose_a, bool tranpose_b, ConvertFP32 functor) {
    int height_c = matrix_a.size() / width_a, width_c = width_b, length = width_a;
    int stride_a_h = width_a, stride_a_w = 1, stride_b_h = width_b, stride_b_w = 1;
    if (tranpose_a) {
        length     = matrix_a.size() / width_a;
        stride_a_w = height_c = width_a;
        stride_a_h            = 1;
    }
    if (tranpose_b) {
        width_c = matrix_b.size() / width_b;
        length = stride_b_w = width_b;
        stride_b_h          = 1;
    }
    matrix_c.resize(height_c * width_c);
    for (int h = 0; h < height_c; ++h) {
        for (int w = 0; w < width_c; ++w) {
            float result = 0;
            for (int i = 0; i < length; ++i) {
                result += functor(matrix_a[h * stride_a_h + i * stride_a_w]) * functor(matrix_b[i * stride_b_h + w * stride_b_w]);
            }
            matrix_c[h * width_c + w] = functor(result);
        }
    }
}
static int randomCreate(int i) {
    i = i + 1023;
    i = (i * 19) % 17;
    i = (i * 23) % 31;
    i = (i * 37) % 41;
    i = (i * 43) % 255;
    return i;
}
using namespace MNN::Express;
class MatMulCommonTest : public MNNTestCase {
public:
    virtual ~MatMulCommonTest() = default;

protected:
    static bool test(MNNForwardType type, const std::string& device_name, const std::string& test_op_name, int height_a,
                     int width_a, int height_b, int width_b, bool tranpose_a, bool tranpose_b,  int precision, bool bConst = false) {
        auto input_a = _Input({height_a, width_a}, NCHW);
        auto input_b = _Input({height_b, width_b}, NCHW);
        vector<float> data_a, data_b, data_c;
        for (int i = 0; i < height_a * width_a; ++i) {
            auto c = randomCreate(i);
            data_a.push_back((float)c / 255.f);
        }
        for (int i = 0; i < height_b * width_b; ++i) {
            auto c = randomCreate(10 - i);
            data_b.push_back((float)c / 255.f);
        }
        reference_matmul(data_a, data_b, data_c, width_a, width_b, tranpose_a, tranpose_b, FP32Converter[precision]);
        ::memcpy(input_a->writeMap<float>(), data_a.data(), data_a.size() * sizeof(float));
        ::memcpy(input_b->writeMap<float>(), data_b.data(), data_b.size() * sizeof(float));
        VARP output;
        if (bConst) {
            VARP A, B;
            if (tranpose_a) {
                A = _Transpose(input_a, {1, 0});
            } else {
                A = input_a;
            }
            //A.fix(VARP::INPUT);
            A = _Unsqueeze(A, {2, 3});
            if (tranpose_b) {
                B = input_b;
            } else {
                B = _Transpose(input_b, {1, 0});
            }
            A = _Convert(A, NC4HW4);
            std::vector<float> weight(B->getInfo()->size);
            ::memcpy(weight.data(), B->readMap<float>(), weight.size() * sizeof(float));
            std::vector<float> bias(B->getInfo()->dim[0]);
            ::memset(bias.data(), 0, bias.size() * sizeof(float));
            auto channelInput = A->getInfo()->dim[1];
            auto channelOutput = B->getInfo()->dim[0];
            auto convOutput = _Conv(std::move(weight), std::move(bias), A, {channelInput, channelOutput}, {1, 1});
            output = _Convert(convOutput, NCHW);
        } else {
            output  = _MatMul(input_a, input_b, tranpose_a, tranpose_b);
        }
        auto outputPtr = output->readMap<float>();
        if (!checkVectorByRelativeError<float>(outputPtr, data_c.data(), data_c.size(), 5e-3)) {
            MNN_ERROR("%s: %d x %d - %d x %d -> %d, %d , transpose: %d, %d, test failed!\n", test_op_name.c_str(),
                      width_a, height_a, width_b, height_b, output->getInfo()->dim[1], output->getInfo()->dim[0],
                      tranpose_a, tranpose_b);
            for (int i = 0; i < data_c.size(); ++i) {
                MNN_PRINT("Correct: %f - Compute: %f\n", data_c[i], outputPtr[i]);
            }
            return false;
        }
        return true;
    }
};

class MatMulTest : public MatMulCommonTest {
public:
    virtual ~MatMulTest() = default;

protected:
    static bool test(MNNForwardType type, const std::string& device_name, int precision) {
        {
            bool succ = MatMulCommonTest::test(MNN_FORWARD_CPU, "device_name", "MatMul", 6, 1, 1,
                                               6, true, true, precision, false);
            if (!succ) {
                return false;
            }
        }

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
                            bool succ = MatMulCommonTest::test(type, device_name, "MatMul", height_a, width_a, height_b,
                                                               width_b, tranpose_a != 0, tranpose_b != 0, precision);
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
    virtual bool run(int precision) {
        return MatMulTest::test(MNN_FORWARD_CPU, "CPU", precision);
    }
};

class MatMulTestBConst : public MatMulTest {
public:
    virtual ~MatMulTestBConst() = default;

protected:
    virtual bool run(int precision) {
        {
            // Test avoid crash
            int e = 5, l = 5, h = 4;
            // Use Conv1x1 instead of MatMul
            auto x0 = MNN::Express::_Input({1, l, e, 1}, NC4HW4, halide_type_of<float>());
            auto y = _Conv(0.0f, 0.0f, x0, {l, h}, {1, 1});
            Variable::prepareCompute({y});
            //Prepare
            x0->writeMap<float>();
            y->readMap<float>();
        }
        {
            bool succ = MatMulCommonTest::test(MNN_FORWARD_CPU, "device_name", "MatMul", 2, 2, 2,
                                               1, true, false, precision, true);
            if (!succ) {
                return false;
            }
        }
        {
            int height_c = 1;
            int width_c = 64;
            int length = 3;
            int height_a = height_c, height_b = length, width_a = length, width_b = width_c;
            bool succ = MatMulCommonTest::test(MNN_FORWARD_CPU, "device_name", "MatMul",height_a, width_a, height_b, width_b, false, false, precision, true);
            if (!succ) {
                return false;
            }
        }
        for (int height_c = 1; height_c <= 48; height_c+=3) {
            for (int width_c = 1; width_c <= 48; width_c+=5) {
                for (int length = 1; length <= 20; length+=7) {
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
                            bool succ = MatMulCommonTest::test(MNN_FORWARD_CPU, "device_name", "MatMul", height_a, width_a, height_b,
                                                               width_b, tranpose_a != 0, tranpose_b != 0, precision, true);
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

MNNTestSuiteRegister(MatMulTestOnCPU, "op/matmul");
MNNTestSuiteRegister(MatMulTestBConst, "op/matmulBConst");
