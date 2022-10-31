//
//  MatrixBandPartGradTest.cpp
//  MNNTests
//
//  Created by MNN on 2022/08/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../tools/train/source/grad/OpGrad.hpp"

using namespace MNN;
using namespace MNN::Express;

class MatrixBandPartGradTest : public MNNTestCase {
public:
    char name[20] = "MatrixBandPart";
    virtual ~MatrixBandPartGradTest() = default;

    virtual bool run(int precision) {
        std::vector<int> shape = {3, 2, 3, 2};
        const int len = shape[0] * shape[1] * shape[2] * shape[3];
        auto input = _Input(shape, NCHW);
        const float inpudata[] = {  -0.94003415, -0.03546342, -0.01028545,  1.2092209 ,  1.5427123 ,
                                    1.0838836 ,  0.5939991 ,  2.017224  ,  0.2702435 ,  1.1762271 ,
                                    -0.95548075, -0.12556452, -1.4086435 , -0.13807571, -0.23514274,
                                    -0.5025484 ,  0.93871444, -0.5169497 , -1.5226837 , -0.8545326 ,
                                    0.340934  ,  0.25505793,  1.7961069 , -0.7955173 , -0.16109313,
                                    1.3417882 ,  0.9252207 , -0.69964254, -0.5392309 ,  0.4769467 ,
                                    -0.33865267, -0.5568984 , -0.53030866, -0.07422069, -0.7438325 ,
                                    -0.06075661};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, len * sizeof(float));

        auto lower = _Scalar<int>(0);
        auto upper = _Scalar<int>(1);
        auto output = _MatrixBandPart(input, lower, upper);

        auto opExpr = output->expr().first;
        auto grad = OpGrad::get(opExpr->get()->type());
        float outputDiff[len] = {   0.92951214, -1.3656238 ,  0.9058341 ,  0.21897921, -0.5062561 ,
                                    0.29703847, -0.5324379 ,  0.8826049 , -0.9250548 ,  1.8164085 ,
                                    -1.7761891 ,  1.2291343 ,  0.45859334,  0.09624046, -0.8051032 ,
                                    0.446291  ,  0.9178219 , -0.7392022 ,  2.31639   , -0.8006644 ,
                                    0.5834905 ,  1.5046587 , -0.11566874, -2.449344  , -1.2720072 ,
                                    -0.55631214, -0.12848197, -1.2433224 , -0.46224716,  0.57611173,
                                    -1.0455252 ,  1.1562699 , -1.2612194 ,  0.46669045,  0.38025302,
                                    -0.70845205};
        auto inputGrad = grad->onGrad(opExpr, {_Const(outputDiff, shape)});

        const std::vector<float> expectedOutput = { 0.92951214, -1.3656238 ,  0.        ,  0.21897921, -0.        ,
                                                    0.        , -0.5324379 ,  0.8826049 , -0.        ,  1.8164085 ,
                                                    -0.        ,  0.        ,  0.45859334,  0.09624046, -0.        ,
                                                    0.446291  ,  0.        , -0.        ,  2.31639   , -0.8006644 ,
                                                    0.        ,  1.5046587 , -0.        , -0.        , -1.2720072 ,
                                                    -0.55631214, -0.        , -1.2433224 , -0.        ,  0.        ,
                                                    -1.0455252 ,  1.1562699 , -0.        ,  0.46669045,  0.        ,
                                                    -0.};
        auto gotOutput = inputGrad[0]->readMap<float>();

        for (int i = 0; i < len; ++i) {
            auto diff = ::fabsf(gotOutput[i] - expectedOutput[i]);
            if (diff > 0.0001) {
                MNN_ERROR("%s grad test failed, expected: %f, but got: %f!\n", name, expectedOutput[i], gotOutput[i]);
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(MatrixBandPartGradTest, "grad/matrix_band_part");
