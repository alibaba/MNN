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
using namespace std;

class BinaryTestCommon : public MNNTestCase {
protected:
    template<typename Tin, typename Tout>
    bool test(VARP (*opFunc)(VARP, VARP), string name, Tout threshold,
              const vector<Tin>& data_x, const vector<Tin>& data_y, const vector<Tout>& data_out,
              const vector<int>& shape_x, const vector<int>& shape_y, const vector<int>& shape_out) {
        int size_x = 1, size_y = 1, size_out = 1;
        for (int i = 0; i < shape_x.size(); ++i) {
            size_x *= shape_x[i];
        }
        for (int i = 0; i < shape_y.size(); ++i) {
            size_y *= shape_y[i];
        }
        for (int i = 0; i < shape_y.size(); ++i) {
            size_out *= shape_out[i];
        }

        auto input_x = _Input(shape_x, NCHW, halide_type_of<Tin>());
        auto input_y = _Input(shape_y, NCHW, halide_type_of<Tin>());
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        auto ptr_x = input_x->template writeMap<Tin>();
        auto ptr_y = input_y->template writeMap<Tin>();
        memcpy(ptr_x, data_x.data(), size_x * sizeof(Tin));
        memcpy(ptr_y, data_y.data(), size_y * sizeof(Tin));
        input_x->unMap();
        input_y->unMap();
        auto output = opFunc(input_x, input_y);
        auto gotOutput = output->template readMap<Tout>();

        auto shape_got = output->getInfo()->dim;
        if (shape_got.size() != shape_out.size()) {
            MNN_ERROR("%s shape compute error!\n", name.c_str());
            return false;
        }
        for (int i = 0; i < shape_got.size(); i++) {
            if (shape_got[i] != shape_out[i]) {
                MNN_ERROR("%s shape compute error!\n", name.c_str());
                return false;
            }
        }

        if (!checkVectorByRelativeError<Tout>(gotOutput, data_out.data(), size_out, threshold)) {
            MNN_ERROR("%s test failed!\n", name.c_str());
            return false;
        }
        return true;
    }
};

class AddTest : public BinaryTestCommon {
public:
    virtual ~AddTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Add, "AddTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.0, 0.0, 0.0},
                    {4}, {4}, {4});
    }
};

class SubtractTest : public BinaryTestCommon {
public:
    virtual ~SubtractTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Subtract, "SubtractTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {-2.0, -4.0, -6.0, -8.0},
                    {4}, {4}, {4});
    }
};
class MultiplyTest : public BinaryTestCommon {
public:
    virtual ~MultiplyTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Multiply, "MultiplyTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {-1.0, -4.0, -9.0, -16.0},
                    {4}, {4}, {4});
    }
};
class DivideTest : public BinaryTestCommon {
public:
    virtual ~DivideTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Divide, "DivideTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {2.0, 4.0, 6.0, 8.0}, {-0.5, -0.5, -0.5, -0.5},
                    {4}, {4}, {4});
    }
};
class PowTest : public BinaryTestCommon {
public:
    virtual ~PowTest() = default;
    virtual bool run(int precision) {
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 10;
        return test<float, float>(_Pow, "PowTest", 0.01 * errorScale,
                    {-1.0, -2.0, -3.0, -4.0}, {2.0, 4.0, 6.0, 4.0}, {1.0, 16.0, 729.0, 256.0},
                    {4}, {4}, {4});
    }
};
class MinimumTest : public BinaryTestCommon {
public:
    virtual ~MinimumTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Minimum, "MinimumTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {-1.0, -2.0, -3.0, -4.0},
                    {4}, {4}, {4});
    }
};
class MaximumTest : public BinaryTestCommon {
public:
    virtual ~MaximumTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_Maximum, "MaximumTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {2.0, 4.0, 6.0, 8.0}, {2.0, 4.0, 6.0, 8.0},
                    {4}, {4}, {4});
    }
};
class BiasAddTest : public BinaryTestCommon {
public:
    virtual ~BiasAddTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_BiasAdd, "BiasAddTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0},
                    {1.0, 2.0},
                    {0.0, 0.0, -2.0, -2.0, -4.0, -4.0, -6.0, -6.0},
                    {4, 2}, {2}, {4, 2});
    }
};
class GreaterTest : public BinaryTestCommon {
public:
    virtual ~GreaterTest() = default;
    virtual bool run(int precision) {
        return test<float, int>(_Greater, "GreaterTest", 0,
                    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    {3.0, 4.0},
                    {0, 0, 0, 0, 1, 1, 1, 1},
                    {4, 2}, {2}, {4, 2});
    }
};
class GreaterEqualTest : public BinaryTestCommon {
public:
    virtual ~GreaterEqualTest() = default;
    virtual bool run(int precision) {
        return test<float, int>(_GreaterEqual, "GreaterEqualTest", 0,
                    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    {3.0, 4.0},
                    {0, 0, 1, 1, 1, 1, 1, 1},
                    {4, 2}, {2}, {4, 2});
    }
};
class LessTest : public BinaryTestCommon {
public:
    virtual ~LessTest() = default;
    virtual bool run(int precision) {
        return test<float, int>(_Less, "LessTest", 0,
                    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    {3.0, 4.0},
                    {1, 1, 0, 0, 0, 0, 0, 0},
                    {4, 2}, {2}, {4, 2});
    }
};
class FloorDivTest : public BinaryTestCommon {
public:
    virtual ~FloorDivTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_FloorDiv, "FloorDivTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.1},
                    {3.0, 4.0},
                    {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0},
                    {4, 2}, {2}, {4, 2});
    }
};
class ModTestInt : public BinaryTestCommon {
public:
    virtual ~ModTestInt() = default;
    virtual bool run(int precision) {
        std::vector<int> x = {
            -4, 7, 5, 4, -7, 8
        };
        std::vector<int> y = {
            2, -3, 8, -2, 3, 5
        };
        std::vector<int> z = {
            0, -2,  5,  0,  2,  3
        };
        return test<int, int>(_Mod, "ModTestFloat", 0,
                              x,y,z, {6}, {6}, {6});
    }
};
class ModTestFloat : public BinaryTestCommon {
public:
    virtual ~ModTestFloat() = default;
    virtual bool run(int precision) {
        std::vector<float> x = {
            1.1f, 2.3f, 3.5f, 4.7f, 5.9f, 6.2f, 7.4f, 8.6f
        };
        std::vector<float> y = {
            0.4f, 0.6f
        };
        std::vector<float> z(x.size());
        for (int i=0; i<2; ++i) {
            for (int j=0; j<4; ++j) {
                z[i + j * 2] = FP32Converter[precision](fmodf(FP32Converter[precision](x[i+j*2]), FP32Converter[precision](y[i])));
            }
        }
        return test<float, float>(_Mod, "ModTestFloat", 0,
                    x,y,z,
                    {4, 2}, {2}, {4, 2});
    }
};
class SquaredDifferenceTest : public BinaryTestCommon {
public:
    virtual ~SquaredDifferenceTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_SquaredDifference, "SquaredDifferenceTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.001},
                    {3.0, 4.0},
                    {16.0, 36.0, 36.0, 64.0, 4.0, 4.0, 16.0, 16.0},
                    {4, 2}, {2}, {4, 2});
    }
};
class EqualTest : public BinaryTestCommon {
public:
    virtual ~EqualTest() = default;
    virtual bool run(int precision) {
        return test<float, int>(_Equal, "EqualTest", 0,
                    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    {3.0, 4.0},
                    {0, 0, 1, 1, 0, 0, 0, 0},
                    {4, 2}, {2}, {4, 2});
    }
};
class LessEqualTest : public BinaryTestCommon {
public:
    virtual ~LessEqualTest() = default;
    virtual bool run(int precision) {
        return test<float, int>(_LessEqual, "LessEqualTest", 0,
                    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    {3.0, 4.0},
                    {1, 1, 1, 1, 0, 0, 0, 0},
                    {4, 2}, {2}, {4, 2});
    }
};
class FloorModTest : public BinaryTestCommon {
public:
    virtual ~FloorModTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_FloorMod, "FloorModTest", 0.01,
                    {-1.0f, -2.0f, -3.0f, -4.0f, 5.0f, 6.0f, 7.0f, 8.1f},
                    {3.0f, 4.0f},
                    {2.0f, 2.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 0.1f},
                    {4, 2}, {2}, {4, 2});
    }
};
class Atan2Test : public BinaryTestCommon {
public:
    virtual ~Atan2Test() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Atan2, "Atan2Test", 0.01,
                    {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0},
                    {3.0, 4.0},
                    {-0.32175055, -0.4636476, -0.7853982, -0.7853982, 1.0303768,   0.98279375, 1.1659045,  1.1071488},
                    {4, 2}, {2}, {4, 2});
    }
};
class LogicalOrTest : public BinaryTestCommon {
public:
    virtual ~LogicalOrTest() = default;
    virtual bool run(int precision) {
        return test<int, int>(_LogicalOr, "LogicalOrTest", 0,
                    {true, false, true, false, false, true, true, false},
                    {true, false},
                    {true, false, true, false, true, true, true, false},
                    {4, 2}, {2}, {4, 2});
    }
};
class NotEqualTest : public BinaryTestCommon {
public:
    virtual ~NotEqualTest() = default;
    virtual bool run(int precision) {
        return test<int, int>(_NotEqual, "NotEqualTest", 0,
                    {true, false, true, false, false, true, true, false},
                    {true, false},
                    {false, false, false, false, true, true, false, false},
                    {4, 2}, {2}, {4, 2});
    }
};
class BitwiseAndTest : public BinaryTestCommon {
public:
    virtual ~BitwiseAndTest() = default;
    virtual bool run(int precision) {
        return test<int, int>(_BitwiseAnd, "BitwiseAndTest", 0,
                    {1, 2, 3, 4, 5, 6, 7, 8},
                    {8, 7, 6, 5, 4, 3, 2, 1},
                    {0, 2, 2, 4, 4, 2, 2, 0},
                    {8}, {8}, {8});
    }
};
class BitwiseOrTest : public BinaryTestCommon {
public:
    virtual ~BitwiseOrTest() = default;
    virtual bool run(int precision) {
        return test<int, int>(_BitwiseOr, "BitwiseOrTest", 0,
                    {1, 2, 3, 4, 5, 6, 7, 8},
                    {8, 7, 6, 5, 4, 3, 2, 1},
                    {9, 7, 7, 5, 5, 7, 7, 9},
                    {8}, {8}, {8});
    }
};
class BitwiseXorTest : public BinaryTestCommon {
public:
    virtual ~BitwiseXorTest() = default;
    virtual bool run(int precision) {
        return test<int, int>(_BitwiseXor, "BitwiseXorTest", 0,
                    {1, 2, 3, 4, 5, 6, 7, 8},
                    {8, 7, 6, 5, 4, 3, 2, 1},
                    {9, 5, 5, 1, 1, 5, 5, 9},
                    {8}, {8}, {8});
    }
};

class BinaryBroadcastShapeTest : public BinaryTestCommon {
public:
    virtual ~BinaryBroadcastShapeTest() = default;
    virtual bool run(int precision) {
        vector<int> data_x(8, 1), data_y(8, 1), data_out(64, 2);
        vector<int> shape_x = {4, 1, 2, 1}, shape_y = {2, 1, 4}, shape_out = {4, 2, 2, 4};
        return test<int, int>(_Add, "BinaryBroadcastShapeTest", 0,
                              data_x, data_y, data_out, shape_x, shape_y, shape_out);
    }
};

class SubtractBroastTest : public BinaryTestCommon {
public:
    virtual ~SubtractBroastTest() = default;
    virtual bool run(int precision) {
        vector<float> data_x(560), data_y(20 * 560), data_out(20 * 560);
        vector<int> shape_x = {560}, shape_y = {1, 20, 560}, shape_out = {1, 20, 560};
        auto func = FP32Converter[precision];
        for (int i = 0; i < 560; ++i) {
            data_x[i]  = func(i / 1000.0f);
        }
        for (int i = 0; i < 560 * 20; ++i) {
            data_y[i]  = func(i / 1000.0f);
        }
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 560; ++j) {
                data_out[j + i * 560] = func(data_x[j] - data_y[j + i * 560]);
            }
        }
        return test<float, float>(_Subtract, "SubtractBroastTest", 0.01,
                                  data_x, data_y, data_out, shape_x, shape_y, shape_out);
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
MNNTestSuiteRegister(ModTestFloat, "op/binary/mod_float");
MNNTestSuiteRegister(ModTestInt, "op/binary/mod_int");
MNNTestSuiteRegister(Atan2Test, "op/binary/atan2");
MNNTestSuiteRegister(LogicalOrTest, "op/binary/logicalor");
MNNTestSuiteRegister(NotEqualTest, "op/binary/notqual");
MNNTestSuiteRegister(SubtractBroastTest, "op/binary/subtractBroastTest");
MNNTestSuiteRegister(BitwiseAndTest, "op/binary/bitwise_and");
MNNTestSuiteRegister(BitwiseOrTest, "op/binary/bitwise_or");
MNNTestSuiteRegister(BitwiseXorTest, "op/binary/bitwise_xor");
