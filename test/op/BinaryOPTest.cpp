//
//  BinaryOPTest.cpp
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

class BinaryBroadcastShapeTest : public MNNTestCase {
public:
    virtual ~BinaryBroadcastShapeTest() = default;
    virtual bool run() {
        auto input_x = _Const(1, {4, 1, 2, 1}, NCHW);
        auto input_y = _Const(1, {2, 1, 4}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        auto output = _Add(input_x, input_y);
        const std::vector<int> expectedOutputShape = {4, 2, 2, 4};
        auto outputSize = output->getInfo()->dim.size();
        if (outputSize != expectedOutputShape.size()) {
            MNN_ERROR("BinaryBroadcastShapeTest shape compute error!\n");
            return false;
        }
        for (int i = 0; i < outputSize; i++) {
            if (output->getInfo()->dim[i] != expectedOutputShape[i]) {
                MNN_ERROR("BinaryBroadcastShapeTest shape compute error!\n");
                return false;
            }
        }
        const std::vector<float> expectedOutput = {
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.};
        auto outputPtr = output->readMap<float>();
        if (!checkVector<float>(outputPtr, expectedOutput.data(), outputSize, 1e-6)) {
            MNN_ERROR("BinaryBroadcastShapeTest compute error!\n");
            return false;
        }
        return true;
    }
};

class AddTest : public MNNTestCase {
public:
    virtual ~AddTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,}, NCHW);
        auto input_y = _Input({4,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0};
        const float data_y[] = {1.0, 2.0, 3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 4 * sizeof(float));
        memcpy(ptr_y, data_y, 4 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Add(input_x, input_y);
        const std::vector<float> expectedOutput = {0.0, 0.0, 0.0, 0.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("AddTest test failed!\n");
            return false;
        }
        return true;
    }
};
class SubtractTest : public MNNTestCase {
public:
    virtual ~SubtractTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,}, NCHW);
        auto input_y = _Input({4,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0};
        const float data_y[] = {1.0, 2.0, 3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 4 * sizeof(float));
        memcpy(ptr_y, data_y, 4 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Subtract(input_x, input_y);
        const std::vector<float> expectedOutput = {-2.0, -4.0, -6.0, -8.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("SubtractTest test failed!\n");
            return false;
        }
        return true;
    }
};
class MultiplyTest : public MNNTestCase {
public:
    virtual ~MultiplyTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,}, NCHW);
        auto input_y = _Input({4,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0};
        const float data_y[] = {1.0, 2.0, 3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 4 * sizeof(float));
        memcpy(ptr_y, data_y, 4 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Multiply(input_x, input_y);
        const std::vector<float> expectedOutput = {-1.0, -4.0, -9.0, -16.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("MultiplyTest test failed!\n");
            return false;
        }
        return true;
    }
};
class DivideTest : public MNNTestCase {
public:
    virtual ~DivideTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,}, NCHW);
        auto input_y = _Input({4,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0};
        const float data_y[] = {2.0, 4.0, 6.0, 8.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 4 * sizeof(float));
        memcpy(ptr_y, data_y, 4 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Divide(input_x, input_y);
        const std::vector<float> expectedOutput = {-0.5, -0.5, -0.5, -0.5};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("DivideTest test failed!\n");
            return false;
        }
        return true;
    }
};
class PowTest : public MNNTestCase {
public:
    virtual ~PowTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,}, NCHW);
        auto input_y = _Input({4,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0};
        const float data_y[] = {2.0, 4.0, 6.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 4 * sizeof(float));
        memcpy(ptr_y, data_y, 4 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Pow(input_x, input_y);
        const std::vector<float> expectedOutput = {1.0, 16.0, 729.0, 256.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("PowTest test failed!\n");
            return false;
        }
        return true;
    }
};
class MinimumTest : public MNNTestCase {
public:
    virtual ~MinimumTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,}, NCHW);
        auto input_y = _Input({4,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0};
        const float data_y[] = {2.0, 4.0, 6.0, 8.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 4 * sizeof(float));
        memcpy(ptr_y, data_y, 4 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Minimum(input_x, input_y);
        const std::vector<float> expectedOutput = {-1.0, -2.0, -3.0, -4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("MinimumTest test failed!\n");
            return false;
        }
        return true;
    }
};
class MaximumTest : public MNNTestCase {
public:
    virtual ~MaximumTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,}, NCHW);
        auto input_y = _Input({4,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0};
        const float data_y[] = {2.0, 4.0, 6.0, 8.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 4 * sizeof(float));
        memcpy(ptr_y, data_y, 4 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Maximum(input_x, input_y);
        const std::vector<float> expectedOutput = {2.0, 4.0, 6.0, 8.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("MaximumTest test failed!\n");
            return false;
        }
        return true;
    }
};
class BiasAddTest : public MNNTestCase {
public:
    virtual ~BiasAddTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0};
        const float data_y[] = {1.0, 2.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _BiasAdd(input_x, input_y);
        const std::vector<float> expectedOutput = {0.0, 0.0, -2.0, -2.0, -4.0, -4.0, -6.0, -6.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
            MNN_ERROR("BiasAddTest test failed!\n");
            return false;
        }
        return true;
    }
};
class GreaterTest : public MNNTestCase {
public:
    virtual ~GreaterTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Greater(input_x, input_y);
        const std::vector<int> expectedOutput = {0, 0, 0, 0, 1, 1, 1, 1};
        auto gotOutput = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("GreaterTest test failed!\n");
            return false;
        }
        return true;
    }
};
class GreaterEqualTest : public MNNTestCase {
public:
    virtual ~GreaterEqualTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _GreaterEqual(input_x, input_y);
        const std::vector<int> expectedOutput = {0, 0, 1, 1, 1, 1, 1, 1};
        auto gotOutput = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("GreaterEqualTest test failed!\n");
            return false;
        }
        return true;
    }
};
class LessTest : public MNNTestCase {
public:
    virtual ~LessTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Less(input_x, input_y);
        const std::vector<int> expectedOutput = {1, 1, 0, 0, 0, 0, 0, 0};
        auto gotOutput = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("LessTest test failed!\n");
            return false;
        }
        return true;
    }
};
class FloorDivTest : public MNNTestCase {
public:
    virtual ~FloorDivTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _FloorDiv(input_x, input_y);
        const std::vector<float> expectedOutput = {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
            MNN_ERROR("FloorDivTest test failed!\n");
            return false;
        }
        return true;
    }
};
class SquaredDifferenceTest : public MNNTestCase {
public:
    virtual ~SquaredDifferenceTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _SquaredDifference(input_x, input_y);
        const std::vector<float> expectedOutput = {16.0, 36.0, 36.0, 64.0, 4.0, 4.0, 16.0, 16.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
            MNN_ERROR("SquaredDifferenceTest test failed!\n");
            return false;
        }
        return true;
    }
};
class EqualTest : public MNNTestCase {
public:
    virtual ~EqualTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Equal(input_x, input_y);
        const std::vector<int> expectedOutput = {0, 0, 1, 1, 0, 0, 0, 0};
        auto gotOutput = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("EqualTest test failed!\n");
            return false;
        }
        return true;
    }
};
class LessEqualTest : public MNNTestCase {
public:
    virtual ~LessEqualTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _LessEqual(input_x, input_y);
        const std::vector<int> expectedOutput = {1, 1, 1, 1, 0, 0, 0, 0};
        auto gotOutput = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("LessEqualTest test failed!\n");
            return false;
        }
        return true;
    }
};
class FloorModTest : public MNNTestCase {
public:
    virtual ~FloorModTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _FloorMod(input_x, input_y);
        const std::vector<float> expectedOutput = {2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
            MNN_ERROR("FloorMod test failed!\n");
            return false;
        }
        return true;
    }
};
class Atan2Test : public MNNTestCase {
public:
    virtual ~Atan2Test() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({2,}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {3.0, 4.0};
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 2 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _Atan2(input_x, input_y);
        const std::vector<float> expectedOutput = {-0.32175055, -0.4636476, -0.7853982, -0.7853982, 1.0303768, 0.98279375, 1.1659045 , 1.1071488};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
            MNN_ERROR("Atan2Test test failed!\n");
            return false;
        }
        return true;
    }
};
class LogicalOrTest : public MNNTestCase {
public:
    virtual ~LogicalOrTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW, halide_type_of<int>());
        auto input_y = _Input({2,}, NCHW, halide_type_of<int>());
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const int data_x[] = {true, false, true, false, false, true, true, false};
        const int data_y[] = {true, false};
        auto ptr_x          = input_x->writeMap<int>();
        auto ptr_y          = input_y->writeMap<int>();
        memcpy(ptr_x, data_x, 8 * sizeof(int));
        memcpy(ptr_y, data_y, 2 * sizeof(int));
        input_x->unMap();
        input_y->unMap();
        auto output = _LogicalOr(input_x, input_y);
        const std::vector<int> expectedOutput = {true, false, true, false, true, true, true, false};
        auto gotOutput = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("LogicalOrTest test failed!\n");
            return false;
        }
        return true;
    }
};
class NotEqualTest : public MNNTestCase {
public:
    virtual ~NotEqualTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW, halide_type_of<int>());
        auto input_y = _Input({2,}, NCHW, halide_type_of<int>());
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const int data_x[] = {true, false, true, false, false, true, true, false};
        const int data_y[] = {true, false};
        auto ptr_x          = input_x->writeMap<int>();
        auto ptr_y          = input_y->writeMap<int>();
        memcpy(ptr_x, data_x, 8 * sizeof(int));
        memcpy(ptr_y, data_y, 2 * sizeof(int));
        input_x->unMap();
        input_y->unMap();
        auto output = _NotEqual(input_x, input_y);
        const std::vector<int> expectedOutput = {false, false, false, false, true, true, false, false};
        auto gotOutput = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("NotEqualTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(BinaryBroadcastShapeTest, "op/binary/broadcastShapeTest");
MNNTestSuiteRegister(AddTest, "op/binary/add");
MNNTestSuiteRegister(SubtractTest, "op/binary/subtract");
MNNTestSuiteRegister(MultiplyTest, "op/binary/multiply");
MNNTestSuiteRegister(DivideTest, "op/binary/divide");
MNNTestSuiteRegister(PowTest, "op/binary/pow");
MNNTestSuiteRegister(MinimumTest, "op/binary/minimum");
MNNTestSuiteRegister(MaximumTest, "op/binary/maximum");
MNNTestSuiteRegister(BiasAddTest, "op/binary/biasadd");
MNNTestSuiteRegister(GreaterTest, "op/binary/greater");
MNNTestSuiteRegister(GreaterEqualTest, "op/binary/greaterequal");
MNNTestSuiteRegister(LessTest, "op/binary/less");
MNNTestSuiteRegister(FloorDivTest, "op/binary/floordiv");
MNNTestSuiteRegister(SquaredDifferenceTest, "op/binary/squareddifference");
MNNTestSuiteRegister(EqualTest, "op/binary/equal");
MNNTestSuiteRegister(LessEqualTest, "op/binary/lessequal");
MNNTestSuiteRegister(FloorModTest, "op/binary/floormod");
MNNTestSuiteRegister(Atan2Test, "op/binary/atan2");
MNNTestSuiteRegister(LogicalOrTest, "op/binary/logicalor");
MNNTestSuiteRegister(NotEqualTest, "op/binary/notqual");
