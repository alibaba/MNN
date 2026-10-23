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

class MatrixAffineExportTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        for (int i = Matrix::kMPersp0; i <= Matrix::kMPersp2; ++i) {
            Matrix m = Matrix::MakeAll(1, 0, 0, 0, 1, 0, 0, 0, 1);
            m.set(i, 0.5f);
            float affine[6] = {101, 102, 103, 104, 105, 106};
            MNNTEST_ASSERT(!m.asAffine(nullptr));
            MNNTEST_ASSERT(!m.asAffine(affine));
            for (int j = 0; j < 6; ++j) {
                MNNTEST_ASSERT(affine[j] == 101 + j);
            }
        }
        Matrix m = Matrix::MakeAll(2, 3, 4, 5, 6, 7, 0, 0, 1);
        const float expected[6] = {2, 5, 3, 6, 4, 7};
        float affine[6] = {};
        MNNTEST_ASSERT(m.asAffine(nullptr));
        MNNTEST_ASSERT(m.asAffine(affine));
        for (int i = 0; i < 6; ++i) {
            MNNTEST_ASSERT(affine[i] == expected[i]);
        }
        return true;
    }
};
MNNTestSuiteRegister(MatrixAffineExportTest, "cv/matrix/affine_export");

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

class MatrixMapRectTest : public MNNTestCase {
    static bool equal(const Rect& a, const Rect& b) {
        return a.fLeft == b.fLeft && a.fTop == b.fTop && a.fRight == b.fRight && a.fBottom == b.fBottom;
    }
public:
    virtual bool run(int precision) {
        const float matrices[][6] = {
            {1, 0, 0, 0, 1, 0}, {1, 0, 7, 0, 1, -5}, {-2, 0, 7, 0, 3, -5},
            {0, -1, 0, 1, 0, 0}, {1, 0.5f, 0, 0, 1, 0}
        };
        const Rect expected[] = {
            {1, 2, 4, 6}, {8, -3, 11, 1}, {-1, 1, 5, 13}, {-6, 1, -2, 4}, {2, 2, 7, 6}
        };
        const bool staysRect[] = {true, true, true, true, false};
        const Rect src{1, 2, 4, 6};
        for (int i = 0; i < 5; ++i) {
            Matrix m;
            const auto& a = matrices[i];
            m.setAll(a[0], a[1], a[2], a[3], a[4], a[5], 0, 0, 1);
            Rect dst{101, 102, 103, 104};
            MNNTEST_ASSERT(m.mapRect(&dst, src) == staysRect[i]);
            MNNTEST_ASSERT(equal(dst, expected[i]));
            Rect inplace = src;
            MNNTEST_ASSERT(m.mapRect(&inplace) == staysRect[i]);
            MNNTEST_ASSERT(equal(inplace, expected[i]));
            MNNTEST_ASSERT(equal(m.mapRect(src), expected[i]));
        }
        return true;
    }
};
MNNTestSuiteRegister(MatrixMapRectTest, "cv/matrix/mapRect");
