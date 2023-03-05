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
        const float inpudata[] = {  -0.94003415f, -0.03546342f, -0.01028545f,  1.2092209f ,  1.5427123f ,
                                    1.0838836f ,  0.5939991f ,  2.017224f  ,  0.2702435f ,  1.1762271f ,
                                    -0.95548075f, -0.12556452f, -1.4086435f , -0.13807571f, -0.23514274f,
                                    -0.5025484f ,  0.93871444f, -0.5169497f , -1.5226837f , -0.8545326f ,
                                    0.340934f  ,  0.25505793f,  1.7961069f , -0.7955173f , -0.16109313f,
                                    1.3417882f ,  0.9252207f , -0.69964254f, -0.5392309f ,  0.4769467f ,
                                    -0.33865267f, -0.5568984f , -0.53030866f, -0.07422069f, -0.7438325f ,
                                    -0.06075661f};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, len * sizeof(float));

        auto lower = _Scalar<int>(0);
        auto upper = _Scalar<int>(1);
        auto output = _MatrixBandPart(input, lower, upper);

        auto opExpr = output->expr().first;
        auto grad = OpGrad::get(opExpr->get()->type());
        const float outputDiff[] = {   0.92951214f, -1.3656238f ,  0.9058341f ,  0.21897921f, -0.5062561f ,
                                    0.29703847f, -0.5324379f ,  0.8826049f , -0.9250548f ,  1.8164085f ,
                                    -1.7761891f ,  1.2291343f ,  0.45859334f,  0.09624046f, -0.8051032f ,
                                    0.446291f  ,  0.9178219f , -0.7392022f ,  2.31639f   , -0.8006644f ,
                                    0.5834905f ,  1.5046587f , -0.11566874f, -2.449344f  , -1.2720072f ,
                                    -0.55631214f, -0.12848197f, -1.2433224f , -0.46224716f,  0.57611173f,
                                    -1.0455252f ,  1.1562699f , -1.2612194f ,  0.46669045f,  0.38025302f,
                                    -0.70845205f};
        auto inputGrad = grad->onGrad(opExpr, {_Const(outputDiff, shape)});

        const std::vector<float> expectedOutput = { 0.92951214f, -1.3656238f ,  0.f        ,  0.21897921f, -0.f        ,
                                                    0.f        , -0.5324379f ,  0.8826049f , -0.f        ,  1.8164085f ,
                                                    -0.f        ,  0.f        ,  0.45859334f,  0.09624046f, -0.f        ,
                                                    0.446291f  ,  0.f        , -0.f        ,  2.31639f   , -0.8006644f ,
                                                    0.f        ,  1.5046587f , -0.f        , -0.f        , -1.2720072f ,
                                                    -0.55631214f, -0.f        , -1.2433224f , -0.f        ,  0.f        ,
                                                    -1.0455252f ,  1.1562699f , -0.f        ,  0.46669045f,  0.f        ,
                                                    -0.f};
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
