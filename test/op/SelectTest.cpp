//
//  SelectTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/05/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <random>

#include <MNN/expr/Expr.hpp>
#include <cmath>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"

using namespace MNN::Express;

#define CHECK_OR_RETURN(expr) \
    if (!(expr)) {            \
        return false;         \
    }

#define CHECK_EQ_OR_RETURN(x, y, i)                         \
    if (x->readMap<float>()[i] != y->readMap<float>()[i]) { \
        return false;                                       \
    }

inline static size_t Size(VARP vaule) {
    return vaule->getInfo()->size;
}

static unsigned int seed = 100;
static std::mt19937 rng(seed);
static std::uniform_real_distribution<double> uniform_dist(0, 1);

template <typename T>
void RandInit(VARP value, T lower, T upper) {
    T* pValue = value->writeMap<T>();
    for (int i = 0; i < Size(value); ++i) {
        pValue[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
    }
}

void RandInitBool(VARP value) {
    int* pValue = value->writeMap<int>();
    for (int i = 0; i < Size(value); ++i) {
        pValue[i] = (uniform_dist(rng) > 0.5f);
    }
}

bool RunSelectAndCheckResult(VARP select, VARP input0, VARP input1, int precision) {
    RandInit<float>(input0, -100.f, 100.f);
    RandInit<float>(input1, -100.f, 100.f);
    RandInitBool(select);

    auto output = _Select(select, input0, input1);

    int iter0 = input0->getInfo()->size == 1 ? 0 : 1;
    int iter1 = input1->getInfo()->size == 1 ? 0 : 1;
    auto outputPtr = output->readMap<float>();
    auto input0Ptr = input0->readMap<float>();
    auto input1Ptr = input1->readMap<float>();
    auto func = FP32Converter[precision];
    for (int i = 0; i < Size(output); ++i) {
        int condition = select->readMap<int>()[0];
        // TODO(houjiang): Correct Select.
        if (Size(select) > i) {
            condition = select->readMap<int>()[i];
        }
        float i0 = input0Ptr[i * iter0];
        float i1 = input1Ptr[i * iter1];
        if (condition) {
            if (!checkVectorByRelativeError(&i0, outputPtr + i, 1, 0.01)) {
                MNN_PRINT("%d, %d - %f - %f - %f\n", i, condition, i0, i1, outputPtr[i]);
                return false;
            }
        } else {
            if (!checkVectorByRelativeError(&i1, outputPtr + i, 1, 0.01)) {
                MNN_PRINT("%d, %d - %f - %f - %f\n", i, condition, i0, i1, outputPtr[i]);
                return false;
            }
        }
    }
    return true;
}

bool SelectTester1D(int N, int precision) {
    auto input0 = _Input({N}, NCHW);
    auto input1 = _Input({N}, NCHW);
    {
        auto select = _Input({N}, NCHW, halide_type_of<int>());
        CHECK_OR_RETURN(RunSelectAndCheckResult(select, input0, input1, precision));
    }
    {
        auto select = _Input({1}, NCHW, halide_type_of<int>());
        CHECK_OR_RETURN(RunSelectAndCheckResult(select, input0, input1, precision));
    }
    return true;
}

bool SelectTester4D(int N, int C, int H, int W, int precision) {
    auto input0 = _Input({N, C, H, W}, NCHW);
    auto input1 = _Input({N, C, H, W}, NCHW);
    {
        auto select = _Input({N, C, H, W}, NCHW, halide_type_of<int>());
        CHECK_OR_RETURN(RunSelectAndCheckResult(select, input0, input1, precision));
    }
    {
        auto select = _Input({1}, NCHW, halide_type_of<int>());
        CHECK_OR_RETURN(RunSelectAndCheckResult(select, input0, input1, precision));
    }
    {
        auto select = _Input({N, C, H, W}, NCHW, halide_type_of<int>());
        auto input0 = _Input({1}, NCHW);
        CHECK_OR_RETURN(RunSelectAndCheckResult(select, input0, input1, precision));
    }
    return true;
}

class SelectTester : public MNNTestCase {
public:
    bool run(int precision) override {
        CHECK_OR_RETURN(SelectTester1D(1, precision));
        CHECK_OR_RETURN(SelectTester1D(2, precision));

        CHECK_OR_RETURN(SelectTester4D(1, 1, 1, 1, precision));
        CHECK_OR_RETURN(SelectTester4D(1, 2, 3, 1, precision));
        CHECK_OR_RETURN(SelectTester4D(2, 3, 4, 5, precision));
        return true;
    }
};

MNNTestSuiteRegister(SelectTester, "op/select");
