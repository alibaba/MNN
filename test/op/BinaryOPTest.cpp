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
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace std;

class BinaryTestCommon : public MNNTestCase {
protected:
    template<typename Tin, typename Tout>
    bool test(VARP (*opFunc)(VARP, VARP), string name, float threshold,
              const vector<Tin>& data_x, const vector<Tin>& data_y, const vector<Tout>& data_out,
              const vector<int>& shape_x, const vector<int>& shape_y, const vector<int>& shape_out, const vector<float> quantScales={-100.f, -100.f, -100.f}, const vector<float> zeroPoints={-100.f, -100.f, -100.f}, Dimensionformat format = NCHW) {
        int size_x = 1, size_y = 1, size_out = 1;
        for (int i = 0; i < shape_x.size(); ++i) {
            size_x *= shape_x[i];
        }
        for (int i = 0; i < shape_y.size(); ++i) {
            size_y *= shape_y[i];
        }
        for (int i = 0; i < shape_out.size(); ++i) {
            size_out *= shape_out[i];
        }
        if (format == NC4HW4 && data_x.size() > size_x) {
            size_x = shape_x[0] * UP_DIV(shape_x[1], 4) * shape_x[2] * shape_x[3] * 4;
            size_y = shape_y[0] * UP_DIV(shape_y[1], 4) * shape_y[2] * shape_y[3] * 4;
        }

        auto input_x = _Input(shape_x, format, halide_type_of<Tin>());
        auto input_y = _Input(shape_y, format, halide_type_of<Tin>());
        input_x->setName("input_x");
        input_y->setName("input_y");
        if (quantScales[0] != -100) { // -100 means invalid scale.
            input_x->writeScaleMap(quantScales[0], zeroPoints[0]);
        }
        if (quantScales[1] != -100) {
            input_y->writeScaleMap(quantScales[1], zeroPoints[1]);
        }
        // set input data
        auto ptr_x = input_x->template writeMap<Tin>();
        auto ptr_y = input_y->template writeMap<Tin>();
        memcpy(ptr_x, data_x.data(), size_x * sizeof(Tin));
        memcpy(ptr_y, data_y.data(), size_y * sizeof(Tin));

        input_x->unMap();
        input_y->unMap();
        auto output = opFunc(input_x, input_y);
        
        if (quantScales[2] != -100){
            output->writeScaleMap(quantScales[2], zeroPoints[2]);
        }
        
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
        if (quantScales[0] > 0) {
            for (int i = 0; i < size_out; ++i) {
                auto error = (int32_t)data_out[i] - (int32_t)gotOutput[i];
                if (error * error > 1) {
                    MNN_PRINT("Error case = %d:\n", i);
                    MNN_PRINT("%s Test error: compute result=%d, right value=%d\n", name.c_str(), (int32_t)gotOutput[i], (int32_t)data_out[i]);
                    return false;
                }
            }
            return true;
        }
        std::vector<Tout> computeOut(size_out);
        std::vector<Tout> targetOut(size_out);
        if (format == NC4HW4) {
            int ob = output->getInfo()->dim[0];
            int oc = output->getInfo()->dim[1];
            int plane = output->getInfo()->dim[2] * output->getInfo()->dim[3];
            for (int b = 0; b < ob; ++b){
                for (int c = 0; c < oc; ++c) {
                    for (int p = 0; p < plane; ++p) {
                        int idx0 = p + c * plane + b * oc * plane;
                        int idx1 = (c % 4) + 4 * p + 4 * plane * b + 4 * plane * ob * (c / 4);
                        computeOut[idx0] = gotOutput[idx1];
                        targetOut[idx0] = data_out[idx1];
                    }
                }
            }
            if (!checkVectorByRelativeError<Tout>(computeOut.data(), targetOut.data(), size_out, threshold)) {
                MNN_ERROR("%s test failed!\n", name.c_str());
                return false;
            }
            return true;
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
        return test<float, float>(MNN::Express::_Add, "AddTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.0, 0.0, 0.0},
                    {4}, {4}, {4});
    }
};

class AddBroastTest : public BinaryTestCommon {
public:
    virtual ~AddBroastTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_Add, "AddBroast", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.0, 0.0, 0.0},
                    {1, 1, 4}, {4}, {1, 1, 4});
    }
};

class AddInt8Test : public BinaryTestCommon {
    public: 
        virtual ~AddInt8Test() = default;
        virtual bool run(int precision) {
        int size = 36;
        std::vector<float> inp1(size, 2);
        vector<float> inp2 = {1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6};
        vector<float> rightResult = {3.1, 4.2, 5.3, 6.6,3.1, 4.2, 5.3, 6.6,3.1, 4.2, 5.3, 6.6,3.1, 4.2, 5.3, 6.6,3.1, 4.2, 5.3, 6.6, 3.1, 4.2, 5.3, 6.6, 3.1, 4.2, 5.3, 6.6, 3.1, 4.2, 5.3, 6.6, 3.1, 4.2, 5.3, 6.6};
        printf("AddInt8 test zeropoint is zero\n");
        bool res = test<float, float>(MNN::Express::_Add, "AddInt8Test", 0.01, inp1, inp2, rightResult, {1}, {size}, {size}, {0.4, 0.036, 0.05176},
                                  {0., 0., 0.});
        printf("AddInt8 test zeropoint is not zero\n");
        res = test<float, float>(MNN::Express::_Add, "AddInt8Test", 0.01, inp1, inp2, rightResult, {1}, {size}, {size}, {0.4, 0.036, 0.05176},
                                  {1., 2., 3.});
        return res;
        }
};

class SubtractTest : public BinaryTestCommon {
public:
    virtual ~SubtractTest() = default;
    virtual bool run(int precision) {
        bool result = test<float, float>(MNN::Express::_Subtract, "SubtractTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {-2.0, -4.0, -6.0, -8.0},
                    {4}, {4}, {4});
        result = result && test<float, float>(MNN::Express::_Subtract, "SubtractTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {-2.0, -4.0, -6.0, -8.0},
                    {4}, {4}, {4}, {0.2, -100, 0.2}, {0, 0, 0});
        return result;
    }
};
class SubtractInt8Test : public BinaryTestCommon {
    public:
        virtual ~SubtractInt8Test() = default;
        virtual bool run(int precision) {
        vector<float> inp1 = {7.0, 28.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6,1.1, 2.2, 3.3, 4.6,1.1, 2.2, 3.3, 4.6}, inp2 = {5.7};
        vector<float> rightResult = {1.3, 22.5, -2.4, -1.1, -4.6, -3.5, -2.4, -1.1, -4.6, -3.5, -2.4,
                                    -1.1, -4.6, -3.5, -2.4, -1.1};
        printf("SubtractInt8 test zeropoint is zero\n");
        bool res = test<float, float>(MNN::Express::_Subtract, "SubtractInt8Test", 0.01, inp1, inp2, rightResult,
                                  {4, 4}, {1}, {4, 4}, {0.4, 0.4, 1.0}, {0., 0., 0.});
        printf("SubtractInt8 test zeropoint is not zero\n");
        res = test<float, float>(MNN::Express::_Subtract, "SubtractInt8Test", 0.01, inp1, inp2, rightResult,
                                  {4, 4}, {1}, {4, 4}, {0.4, 0.4, 1.0}, {1., 2., 3.});
        return res;
        }
};

class MultiplyTest : public BinaryTestCommon {
public:
    virtual ~MultiplyTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_Multiply, "MultiplyTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {-1.0, -4.0, -9.0, -16.0},
                    {4}, {4}, {4});
    }
};
class MultiplyInt8Test : public BinaryTestCommon {
public:
    virtual ~MultiplyInt8Test() = default;
    virtual bool run(int precision) {
        vector<float> inp1 = {1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6,}, inp2 = {5.7, 2.5, 0.25, 0.43, 5.7, 2.5, 0.25, 0.43, 5.7, 2.5, 0.25, 0.43, 5.7, 2.5, 0.25, 0.43};
        vector<float> rightResult = {6.27 , 5.5 , 0.825, 1.978, 6.27 , 5.5 , 0.825, 1.978, 6.27 , 5.5 , 0.825, 1.978, 6.27 , 5.5 , 0.825, 1.978};
        printf("MultiplyInt8 test zeropoint is zero\n");
        bool res = test<float, float>(MNN::Express::_Multiply, "MultiplyInt8Test", 0.01, inp1, inp2, rightResult,
                                  {16}, {16}, {16}, {0.4, 0.4, 0.16}, {0., 0., 0.});
        printf("MultiplyInt8 test zeropoint is not zero\n");
        res = test<float, float>(MNN::Express::_Multiply, "MultiplyInt8Test", 0.01, inp1, inp2, rightResult,
                                  {16}, {16}, {16}, {0.4, 0.4, 0.16}, {1., 2., 3.});
        return res;
    }
};

class DivideTest : public BinaryTestCommon {
public:
    virtual ~DivideTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_Divide, "DivideTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {2.0, 4.0, 6.0, 8.0}, {-0.5, -0.5, -0.5, -0.5},
                    {4}, {4}, {4});
    }
};
class DivideInt8Test : public BinaryTestCommon {
public:
    virtual ~DivideInt8Test() = default;
    virtual bool run(int precision) {
        vector<float> inp1 = {1.1, 2.2, 3.3, 4.6}, inp2 = {5.7, 2.5, 2.6, 1.88};
        vector<float> rightResult = {0.19298,  0.88, 1.269, 2.4468};
        printf("DivedeInt8 test zero point is zero\n");
        bool res = test<float, float>(MNN::Express::_Divide, "DivideInt8Test", 0.01, inp1, inp2, rightResult,
                                  {4}, {4}, {4}, {0.4, 0.4, 1.0}, {0., 0., 0.});
        printf("DivedeInt8 test zero point is not zero\n");
        res = test<float, float>(MNN::Express::_Divide, "DivideInt8Test", 0.01, inp1, inp2, rightResult,
                                  {4}, {4}, {4}, {0.4, 0.4, 1.0}, {0., 2., 0.});
        return res;
    }
};

class PowTest : public BinaryTestCommon {
public:
    virtual ~PowTest() = default;
    virtual bool run(int precision) {
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 10;
        bool result = test<float, float>(MNN::Express::_Pow, "PowTest", 0.01 * errorScale,
                    {-1.0, -2.0, -3.0, -4.0}, {2.0, 4.0, 6.0, 4.0}, {1.0, 16.0, 729.0, 256.0},
                    {4}, {4}, {4});
        result = result && test<float, float>(MNN::Express::_Pow, "PowTest", 0.01 * errorScale,
                    {-1.0, -2.0, -3.0, -4.0}, {2.0, 4.0, 6.0, 4.0}, {1.0, 16.0, 729.0, 256.0},
                    {4}, {4}, {4}, {0.3, 0.3, -100}, {0, 0, 0});
        return result;
    }
};
class PowInt8Test : public BinaryTestCommon {
public:
    virtual ~PowInt8Test() = default;
    virtual bool run(int precision) {
        vector<float> inp1 = {-1.0, -2.0, -3.0, -4.0}, inp2 = {2.0, 4.0, 3, 4.0};
        vector<float> rightResult = {1, 16, -27.0, 256};
        return test<float, float>(MNN::Express::_Pow, "PowInt8Test", 0.01, inp1, inp2, rightResult,
                                  {4}, {4}, {4}, {1.0, 1.0, 3.0}, {0., 0., 0.});
    }
};

class MinimumTest : public BinaryTestCommon {
public:
    virtual ~MinimumTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_Minimum, "MinimumTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0}, {1.0, 2.0, 3.0, 4.0}, {-1.0, -2.0, -3.0, -4.0},
                    {4}, {4}, {4});
    }
};
class MinimumInt8Test : public BinaryTestCommon {
public:
    virtual ~MinimumInt8Test() = default;
    virtual bool run(int precision) {
        vector<float> inp1 = {-1.2, -5.0, 8, 10}, inp2 = {9.3, 3.1, 11.0, 2.9};
        vector<float> rightResult = {-1.2, -5.0, 8, 2.9};
        bool res = test<float, float>(MNN::Express::_Minimum, "MinimumInt8Test", 0.01, inp1, inp2, rightResult,
                                  {4}, {4}, {4}, {0.4, 0.4, 1.0}, {0., 0., 0.});
        res = test<float, float>(MNN::Express::_Minimum, "MinimumInt8Test", 0.01, inp1, inp2, rightResult,
                                  {4}, {4}, {4}, {0.4, 0.4, 1.0}, {1., 2., 3.});
        return res;
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
class MaximumInt8Test : public BinaryTestCommon {
public:
    virtual ~MaximumInt8Test() = default;
    virtual bool run(int precision) {
        vector<float> inp1 = {-1, -5, 8, 10, -1, -5, 8, 10,-1, -5, 8, 10,-1, -5, 8, 10,-1, -5, 8, 10,-1, -5, 8, 10,-1, -5, 8, 10,-1, -5, 8, 10}, inp2 = {9};
        vector<float> rightResult = {9, 9, 9, 10,9, 9, 9, 10,9, 9, 9, 10,9, 9, 9, 10,9, 9, 9, 10,9, 9, 9, 10,9, 9, 9, 10,9, 9, 9, 10,};
        printf("MaximumInt8 test zeropoint is zero\n");
        bool res = test<float, float>(MNN::Express::_Maximum, "MaximumInt8Test", 0.01, inp1, inp2, rightResult,
                                  {32}, {1}, {32}, {0.4, 0.4, 1.0}, {0., 0., 0.});
        printf("MaximumInt8 test zeropoint is not zero\n");
        res = test<float, float>(MNN::Express::_Maximum, "MaximumInt8Test", 0.01, inp1, inp2, rightResult,
                                  {32}, {1}, {32}, {0.4, 0.4, 1.0}, {1., 2., 5.});
        return res;
    }
};

class BiasAddTest : public BinaryTestCommon {
public:
    virtual ~BiasAddTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_BiasAdd, "BiasAddTest", 0.01,
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
        auto res = test<int, int>(MNN::Express::_Greater, "GreaterTest", 0,
                                   {1, 2, 3, 4, 5, 6, 7, 8},
                                   {3, 4},
                                   {0, 0, 0, 0, 1, 1, 1, 1},
                                   {4, 2}, {2}, {4, 2});
        return res && test<float, int>(MNN::Express::_Greater, "GreaterTest", 0,
                    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    {3.0, 4.0},
                    {0, 0, 0, 0, 1, 1, 1, 1},
                    {4, 2}, {2}, {4, 2});;
    }
};
class GreaterEqualTest : public BinaryTestCommon {
public:
    virtual ~GreaterEqualTest() = default;
    virtual bool run(int precision) {
        auto res = test<int, int>(MNN::Express::_GreaterEqual, "GreaterEqualTest", 0,
                                    {1, 2, 3, 4, 5, 6, 7, 8},
                                    {3, 4},
                                    {0, 0, 1, 1, 1, 1, 1, 1},
                                    {4, 2}, {2}, {4, 2});
        return res && test<float, int>(MNN::Express::_GreaterEqual, "GreaterEqualTest", 0,
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
        auto res0 = test<int, int>(MNN::Express::_Less, "LessTest", 0,
                                    {1, 2, 3, 4, 5, 6, 7, 8},
                                    {3, 4},
                                    {1, 1, 0, 0, 0, 0, 0, 0},
                                    {4, 2}, {2}, {4, 2});
        auto res1 = test<int, int>(MNN::Express::_LessEqual, "LessEqualTest", 0,
                                    {1, 2, 3, 4, 5, 6, 7, 8},
                                    {3, 4},
                                    {1, 1, 1, 1, 0, 0, 0, 0},
                                    {4, 2}, {2}, {4, 2});
        return res0 && res1 && test<float, int>(MNN::Express::_Less, "LessTest", 0,
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
        return test<float, float>(MNN::Express::_FloorDiv, "FloorDivTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.1},
                    {3.0, 4.0},
                    {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0},
                    {4, 2}, {2}, {4, 2});
    }
};
class FloorDivInt8Test : public BinaryTestCommon {
public:
    virtual ~FloorDivInt8Test() = default;
    virtual bool run(int precision) {
        vector<float> inp1 = {-3.98, 17.5, 25.4, 6.7}, inp2 = {3};
        vector<float> rightResult = {-2, 5, 8, 2};
        return test<float, float>(MNN::Express::_FloorDiv, "FloorDivInt8Test", 0.01, inp1, inp2, rightResult,
                                  {4}, {1}, {4}, {0.4, 0.4, 1}, {0., 0., 0.});
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
        return test<int, int>(_Mod, "ModTestInt", 0,
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
        return test<float, float>(MNN::Express::_Mod, "ModTestFloat", 0.01,
                    x,y,z,
                    {4, 2}, {2}, {4, 2});
    }
};
class SquaredDifferenceTest : public BinaryTestCommon {
public:
    virtual ~SquaredDifferenceTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_SquaredDifference, "SquaredDifferenceTest", 0.01,
                    {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.001},
                    {3.0, 4.0},
                    {16.0, 36.0, 36.0, 64.0, 4.0, 4.0, 16.0, 16.0},
                    {4, 2}, {2}, {4, 2});
    }
};
class SquaredDifferenceInt8Test : public BinaryTestCommon {
public:
    virtual ~SquaredDifferenceInt8Test() = default;
    virtual bool run(int precision) {
        vector<float> inp1 = {-1, -2, -3, -4, 5, 6, 7, 8, -1, -2, -3, -4, 5, 6, 7, 8, -1, -2, -3, -4, 5, 6, 7, 8, -1, -2, -3, -4, 5, 6, 7, 8}, inp2 = {3};
        vector<float> rightResult = {16, 25, 36, 49, 4, 9, 16, 25, 16, 25, 36, 49, 4, 9, 16, 25, 16, 25, 36, 49, 4, 9, 16, 25, 16, 25, 36, 49, 4, 9, 16, 25};
        printf("SqdInt8 test zeropoint is zero\n");
        bool res = test<float, float>(MNN::Express::_SquaredDifference, "SquaredDifferenceInt8Test", 0.01, inp1, inp2, rightResult,
                                  {8, 4}, {1}, {8, 4}, {1, 1, 1}, {0., 0., 0.});
        printf("SqdInt8 test zeropoint is not zero\n");
        res = test<float, float>(MNN::Express::_SquaredDifference, "SquaredDifferenceInt8Test", 0.01, inp1, inp2, rightResult,
                                  {8, 4}, {1}, {8, 4}, {1, 1, 1}, {2., 0., 1.});
        return res;
    }
};

class EqualTest : public BinaryTestCommon {
public:
    virtual ~EqualTest() = default;
    virtual bool run(int precision) {
        return test<float, int>(MNN::Express::_Equal, "EqualTest", 0,
                    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    {3.0, 4.0},
                    {0, 0, 1, 1, 0, 0, 0, 0},
                    {4, 2}, {2}, {4, 2}) &&
        test<int, int>(MNN::Express::_Equal, "EqualIntTest", 0,
                    {1, 2, 3, 4, 5, 6, 7, 8},
                    {3, 4},
                    {0, 0, 1, 1, 0, 0, 0, 0},
                       {4, 2}, {2}, {4, 2});
        ;
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
                    {4, 2}, {2}, {4, 2})
        &&
        test<int, int>(_LessEqual, "LessEqualIntTest", 0,
                    {1, 2, 3, 4, 5, 6, 7, 8},
                    {3, 4},
                    {1, 1, 1, 1, 0, 0, 0, 0},
                    {4, 2}, {2}, {4, 2})
        ;
    }
};
class FloorModTest : public BinaryTestCommon {
public:
    virtual ~FloorModTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_FloorMod, "FloorModTest", 0.01,
                    {-1.0f, -2.0f, -3.0f, -4.0f, 5.0f, 6.0f, 7.0f, 8.1f},
                    {3.0f, 4.0f},
                    {2.0f, 2.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 0.1f},
                    {4, 2}, {2}, {4, 2});
    }
};
class FloorModInt8Test : public BinaryTestCommon {
public:
    virtual ~FloorModInt8Test() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_FloorMod, "FloorModInt8Test", 0.01,
                    {-1, -3, 5, 7},
                    {3.0f}, {2, 0, 2, 1},
                                  {4}, {1}, {4}, {0.3, 0.3, 0.3}, {0., 0., 0.});
    }
};
class Atan2Test : public BinaryTestCommon {
public:
    virtual ~Atan2Test() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_Atan2, "Atan2Test", 0.01,
                    {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0},
                    {3.0, -4.0},
                    {-0.32175055, -2.67794504, -0.7853982, -2.35619449, 1.0303768, 2.15879893, 1.1659045, 2.03444394},
                    {4, 2}, {2}, {4, 2});
    }
};
class Atan2Int8Test : public BinaryTestCommon {
public:
    virtual ~Atan2Int8Test() = default;
    virtual bool run(int precision) {
        return test<float, float>(MNN::Express::_Atan2, "Atan2Int8Test", 0.01,
                    {-1, -3, 5, 7},
                    {3}, {-1, 0, 2, 1},
                                  {4}, {1}, {4}, {1, 1, 1}, {0., 0., 0.});
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
                    {4, 2}, {2}, {4, 2})
        &&
        test<float, int>(_NotEqual, "NotEqualTest", 0,
                    {1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0},
                    {1.0, 0.0},
                    {false, false, false, false, true, true, false, false},
                    {4, 2}, {2}, {4, 2})
        ;
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

class BinaryReluTest : public BinaryTestCommon {
public:
    virtual ~BinaryReluTest() = default;
    virtual bool run(int precision) {
        std::vector<float> input0_data = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        std::vector<float> input1_data = {
            -2.0, 2.0, -4.0,
            4.0, 5.0, -8.0,
            7.0, -18.0, 9.0
        };
        std::vector<float> output_data = {
            0.0, 4.0, 0.0,
            8.0, 10.0, 0.0,
            14.0, 0.0, 18.0
        };
        auto input_0 = _Input({1, 1, 3, 3}, NCHW, halide_type_of<float>());
        auto input_1 = _Input({1, 1, 3, 3}, NCHW, halide_type_of<float>());
        ::memcpy(input_0->writeMap<float>(), input0_data.data(), input0_data.size() * sizeof(float));
        ::memcpy(input_1->writeMap<float>(), input1_data.data(), input1_data.size() * sizeof(float));

        std::unique_ptr<OpT> binaryOp(new OpT);
        binaryOp->type = OpType_BinaryOp;
        binaryOp->main.type = OpParameter_BinaryOp;

        binaryOp->main.value = new BinaryOpT;
        binaryOp->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
        binaryOp->main.AsBinaryOp()->activationType = 1;// Do Relu

        auto output = Variable::create(Expr::create(binaryOp.get(), {input_0, input_1}, 1));
        auto getOutput = output->readMap<float>();
        if (!checkVectorByRelativeError<float>(getOutput, output_data.data(), output_data.size(), 0.001)) {
            MNN_ERROR("Binary-Relu fuse test failed!\n");
            return false;
        }
        return true;
    }
};


class BinaryBroadcastShapeTest : public BinaryTestCommon {
public:
    virtual ~BinaryBroadcastShapeTest() = default;
    virtual bool run(int precision) {
        vector<int> data_x(8, 1), data_y(8, 1), data_out(64, 2);
        vector<int> shape_x = {4, 1, 2, 1}, shape_y = {2, 1, 4}, shape_out = {4, 2, 2, 4};
        return test<int, int>(MNN::Express::_Add, "BinaryBroadcastShapeTest", 0,
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
        return test<float, float>(MNN::Express::_Subtract, "SubtractBroastTest", 0.01,
                                  data_x, data_y, data_out, shape_x, shape_y, shape_out);
    }
};

class AddC4Test : public BinaryTestCommon {
public:
    virtual ~AddC4Test() = default;
    virtual bool run(int precision) {
        {
            vector<float> inp2 = {1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6, 1.1, 2.2, 3.3, 4.6}, inp1 = {2};
            vector<float> rightResult = {3.1, 4.2, 5.3, 6.6,3.1, 4.2, 5.3, 6.6,3.1, 4.2, 5.3, 6.6,3.1, 4.2, 5.3, 6.6,3.1, 4.2, 5.3, 6.6, 3.1, 4.2, 5.3, 6.6, 3.1, 4.2, 5.3, 6.6, 3.1, 4.2, 5.3, 6.6};
            bool res = test<float, float>(MNN::Express::_Add, "AddInt8C4Test", 0.01, inp1, inp2, rightResult, {1, 1, 1, 1}, {1, 32, 1, 1}, {1, 32, 1, 1}, {0.4, 0.4, 1.0},
                                      {1., 2., 3.}, NC4HW4);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
        }
        std::vector<float> i1 = {
            -1.0, -2.0, 0.f, 0.f,
            -3.0, -4.0, 0.f, 0.f,
            -5.0, -6.0, 0.f, 0.f,
            -7.0, -8.0, 0.f, 0.f
        };
        std::vector<float> i0 = {
            1.0f, 0.0f, 0.f, 0.f
        };
        std::vector<float> i2 = {
            0.0, -1.0, 0.f, 0.f,
            -2.0, -3.0, 0.f, 0.f,
            -4.0, -5.0, 0.f, 0.f,
            -6.0, -7.0, 0.f, 0.f
        };
        return test<float, float>(MNN::Express::_BiasAdd, "AddC4FloatTest", 0.01,
                    i0, i1, i2,
                                  {1, 1, 1, 1}, {4, 2, 1, 1}, {4, 2, 1, 1}, {-100.f, -100.f, -100.f}, {-100.f, -100.f, -100.f}, NC4HW4);

    }
};

// Float32 OpTest.
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
MNNTestSuiteRegister(BinaryReluTest, "op/binary/fuse_relu");
// Int8 OpTest.
MNNTestSuiteRegister(AddInt8Test, "op/binary/addInt8");
MNNTestSuiteRegister(SubtractInt8Test, "op/binary/subtractInt8");
MNNTestSuiteRegister(MultiplyInt8Test, "op/binary/multiplyInt8");
MNNTestSuiteRegister(DivideInt8Test, "op/binary/divideInt8");
MNNTestSuiteRegister(PowInt8Test, "op/binary/powInt8");
MNNTestSuiteRegister(MinimumInt8Test, "op/binary/minimumInt8");
MNNTestSuiteRegister(MaximumInt8Test, "op/binary/maximumInt8");
MNNTestSuiteRegister(FloorDivInt8Test, "op/binary/floordivInt8");
MNNTestSuiteRegister(FloorModInt8Test, "op/binary/floormodInt8");
MNNTestSuiteRegister(Atan2Int8Test, "op/binary/atan2Int8");
MNNTestSuiteRegister(SquaredDifferenceInt8Test, "op/binary/sqdInt8");

MNNTestSuiteRegister(AddC4Test, "op/binary/addC4");
MNNTestSuiteRegister(AddBroastTest, "op/binary/AddBroast");

