//
//  ExtractTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "../tools/train/source/nn/NN.hpp"
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"
using namespace MNN;

class ExtractTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        // Make Net
        auto i0 = MNN::Express::_Input({}, MNN::Express::NHWC, halide_type_of<int>());
        auto i1 = MNN::Express::_Input({}, MNN::Express::NHWC, halide_type_of<int>());
        auto i2 = MNN::Express::_Input({}, MNN::Express::NHWC, halide_type_of<int>());
        auto i3 = MNN::Express::_Input({}, MNN::Express::NHWC, halide_type_of<int>());
        auto i4 = MNN::Express::_Input({}, MNN::Express::NHWC, halide_type_of<int>());
        int f0 = 1;
        int f1 = 3;
        int f2 = 4;
        int f3 = 5;
        int f4 = 8;

        auto y0 = i0 + i1; auto fy0 = f0 + f1;
        auto y1 = i1 * i2; auto fy1 = f1 * f2;
        auto y2 = i3 - i0; auto fy2 = f3 - f0;
        auto y3 = i4 / i2; auto fy3 = f4 / f2;
        
        auto z0 = y0 * y1 - y2 * y3;
        auto fz0 = fy0 * fy1 - fy2 * fy3;
        
        std::shared_ptr<MNN::Express::Module> m(MNN::Express::NN::extract({i0, i1, i2, i3, i4}, {z0}, true), [](void* ptr) {
            MNN::Express::Module::destroy((MNN::Express::Module*)ptr);
        });
        auto x0 = MNN::Express::_Scalar(f0);
        auto x1 = MNN::Express::_Scalar(f1);
        auto x2 = MNN::Express::_Scalar(f2);
        auto x3 = MNN::Express::_Scalar(f3);
        auto x4 = MNN::Express::_Scalar(f4);
        
        auto xz = m->onForward({x0, x1, x2, x3, x4})[0]->readMap<int>()[0];
        if (xz != fz0) {
            MNN_ERROR("Compute %d != %d\n", xz, fz0);
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ExtractTest, "nn/Extract");
