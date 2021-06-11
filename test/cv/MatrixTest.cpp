//
//  MatrixTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/Matrix.h>
#include "MNNTestSuite.h"

using namespace MNN::CV;

class MatrixScaleTest : public MNNTestCase {
public:
    virtual ~MatrixScaleTest() = default;
    virtual bool run(int precision) {
        Matrix m;
        m.setRectToRect(Rect{20, 21, 40, 41}, Rect{0, 0, 100, 100}, Matrix::kCenter_ScaleToFit);
        Point p;
        m.invert(&m);
        m.mapXY(50, 50, &p);

        MNNTEST_ASSERT(p.fX == 30);
        MNNTEST_ASSERT(p.fY == 31);
        return true;
    }
};
MNNTestSuiteRegister(MatrixScaleTest, "cv/matrix/scale");
