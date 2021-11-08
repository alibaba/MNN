//
//  MatrixBandTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*
 Test Case From https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/matrix-band-part
 */
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
using namespace MNN::Express;

class MatrixBandTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::unique_ptr<MNN::OpT> MatrixBandOp(new MNN::OpT);
        MatrixBandOp->type        = MNN::OpType_MatrixBandPart;
        auto matrix               = _Input({4, 4}, NHWC, halide_type_of<float>());
        auto lower                = _Input({}, NHWC, halide_type_of<int32_t>());
        auto upper                = _Input({}, NHWC, halide_type_of<int32_t>());
        auto y                    = Variable::create(Expr::create(MatrixBandOp.get(), {matrix, lower, upper}));
        std::vector<float> values = {0.0f,  1.0f,  2.0f, 3.0f, -1.0f, 0.0f,  1.0f,  2.0f,
                                     -2.0f, -1.0f, 0.0f, 1.0f, -3.0f, -2.0f, -1.0f, 0.0f};
        ::memcpy(matrix->writeMap<float>(), values.data(), values.size() * sizeof(float));
        {
            lower->writeMap<int>()[0] = 1;
            upper->writeMap<int>()[0] = -1;
            {
                auto yPtr = y->readMap<float>();
                for (int h = 0; h < 4; ++h) {
                    for (int w = 0; w < 4; ++w) {
                        auto computed = yPtr[4 * h + w];
                        auto expected = 0.0f;
                        if (h - w <= 1) {
                            expected = values[4 * h + w];
                        }
                        if (computed != expected) {
                            FUNC_PRINT(1);
                            return false;
                        }
                    }
                }
            }
        }
        {
            lower->writeMap<int>()[0] = 2;
            upper->writeMap<int>()[0] = 1;
            {
                auto yPtr = y->readMap<float>();
                for (int h = 0; h < 4; ++h) {
                    for (int w = 0; w < 4; ++w) {
                        auto computed = yPtr[4 * h + w];
                        auto expected = 0.0f;
                        if ((h - w) <= 2 && (w - h) <= 1) {
                            expected = values[4 * h + w];
                        }
                        if (computed != expected) {
                            FUNC_PRINT(1);
                            return false;
                        }
                    }
                }
            }
        }
        {
            matrix->resize({3, 5, 5});
            auto matrixPtr = matrix->writeMap<float>();
            for (int i = 0; i < matrix->getInfo()->size; ++i) {
                matrixPtr[i] = (float)i;
            }
            lower->writeMap<int>()[0] = 2;
            upper->writeMap<int>()[0] = 1;
            auto yPtr                 = y->readMap<float>();
            for (int z = 0; z < 3; ++z) {
                for (int h = 0; h < 5; ++h) {
                    for (int w = 0; w < 5; ++w) {
                        auto index    = w + 5 * h + 5 * 5 * z;
                        auto computed = yPtr[index];
                        auto expected = 0.0f;
                        if ((h - w) <= 2 && (w - h) <= 1) {
                            expected = (float)(index);
                        }
                        if (computed != expected) {
                            FUNC_PRINT(1);
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(MatrixBandTest, "expr/MatrixBand");
