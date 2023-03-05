//
//  InterpGrad.cpp
//  MNN
//
//  Created by MNN on 2019/12/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "../source/geometry/ConvertUtils.hpp"
#include <vector>
#include <math.h>

using namespace std;
using namespace MNN;
using namespace MNN::Express;


static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

// F = -0.75
static std::vector<float> CubicInterpolation2(float t) {
    float b0 = 1.0f - 2.25f * t * t + 1.25f * t * t * t;
    float c0 = 1.0f - 2.25f * (1.0f - t) * (1.0f - t) + 1.25f * (1.0f - t) * (1.0f - t) * (1.0f - t);
    float t_a = 1.0f + t;
    float t_d = 2.0f - t;
    float a0 = 3.0f - 6.0f * (t_a) + 5.0f * 0.75 * t_a * t_a - 0.75f * t_a * t_a * t_a;
    float d0 = 3.0f - 6.0f * (t_d) + 5.0f * 0.75 * t_d * t_d - 0.75f * t_d * t_d * t_d;
    
    return {a0, b0, c0, d0};
}

// F = -0.75
static VARPS CubicInterpolation2(VARP t, VARPS &fs, int axis) {
    auto sevenFive = fs[0];
    auto one = fs[1];
    auto oneTwoFive = fs[2];
    auto two = fs[3];
    auto twoTwoFive = fs[4];
    auto three = fs[5];
    auto five = fs[6];
    auto six = fs[7];

    auto st = _Square(t);
    auto oneMinusT = one - t;
    auto sOneMinusT = _Square(oneMinusT);

    auto b0 = one - twoTwoFive * st + oneTwoFive * t * st;
    auto c0 = one - twoTwoFive * sOneMinusT + oneTwoFive * oneMinusT * sOneMinusT;
    auto ta = one + t;
    auto sTa = _Square(ta);
    auto td = two - t;
    auto sTd = _Square(td);
    auto a0 = three - six * ta + five * sevenFive * sTa - sevenFive * ta * sTa;
    auto d0 = three - six * td + five * sevenFive * sTd - sevenFive * td * sTd;

    return {a0, b0, c0, d0};
}

class InterpGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto op = expr->get();
        auto originInterpParam = op->main_as_Interp();
        auto inputs = expr->inputs();
        auto inputInfo = expr->inputs()[0]->getInfo();
        MNN_ASSERT(nullptr != inputInfo && inputInfo->dim.size() == 4);
        auto outputDiff = backwardOutput[0];
        outputDiff = _Convert(outputDiff, NCHW);
        auto diffInfo = outputDiff->getInfo();
        std::vector<VARP> res{nullptr};

        const int inB = inputInfo->dim[0];
        const int inC = inputInfo->dim[1];
        const int inH = inputInfo->dim[2];
        const int inW = inputInfo->dim[3];

        const int outB = diffInfo->dim[0];
        const int outC = diffInfo->dim[1];
        const int outH = diffInfo->dim[2];
        const int outW = diffInfo->dim[3];

        int resizeType;
        bool alignCorners;
        InterpInfo info;
        
        if (OpType_Resize == op->type()) {
            alignCorners = false;
            resizeType   = 2; // Bilinear
            info.widthScale = (float)inW / outW;
            info.heightScale = (float)inH / outH;
        } else {
            MNN_ASSERT(OpType_Interp == op->type());
            resizeType   = originInterpParam->resizeType();
            alignCorners = originInterpParam->alignCorners();
            bool computeScale = true;
            if (inputs.size() > 1 && inputs[1]->getInfo()->type.code == halide_type_float) {
                computeScale = false;
                auto scalePtr = inputs[1]->readMap<float>();
                info.heightScale = 1.0f / scalePtr[2];
                if (inputs[0]->getInfo()->dim.size() >= 4) {
                    info.widthScale = 1.0f / scalePtr[3];
                }
            }
            MNN_ASSERT(nullptr != inputInfo && inputInfo->dim.size() == 4);
            const int defaultDepth = 10;
            _ConverterInterp(originInterpParam, &info, inW, inH, defaultDepth, outW, outH, defaultDepth, computeScale);
        }

        // nearest, nearest_round
        if (resizeType == 1 || resizeType == 4) {
            const float scaleH = info.heightScale;
            const float scaleW = info.widthScale;

            vector<int> vecH, vecW;
            for (int i = 0; i < outH; i++) {
                vecH.push_back(i);
            }
            for (int i = 0; i < outW; i++) {
                vecW.push_back(i);
            }

            auto varH = _Const((void*)vecH.data(), {outH, 1, 1}, NCHW, halide_type_of<int>());
            auto varW = _Const((void*)vecW.data(), {1, outW, 1}, NCHW, halide_type_of<int>());

            if (resizeType == 1) {
                varH = _Cast<int>(_Floor(_Scalar<float>(scaleH) * _Cast<float>(varH)));
                varW = _Cast<int>(_Floor(_Scalar<float>(scaleW) * _Cast<float>(varW)));
            } else {
                varH = _Cast<int>(_Round(_Scalar<float>(scaleH) * (_Cast<float>(varH) + _Scalar<float>(0.5f)) - _Scalar<float>(0.5f)));
                varW = _Cast<int>(_Round(_Scalar<float>(scaleW) * (_Cast<float>(varW) + _Scalar<float>(0.5f)) - _Scalar<float>(0.5f)));
            }
            varH = Express::_Maximum(varH, _Scalar<int>(0));
            varH = Express::_Minimum(varH, _Scalar<int>(inH - 1));
            varW = Express::_Maximum(varW, _Scalar<int>(0));
            varW = Express::_Minimum(varW, _Scalar<int>(inW - 1));

            auto expandH = varH * _Cast<int>(_Const(1.0f, {1, outW, 1}));
            auto expandW = varW * _Cast<int>(_Const(1.0f, {outH, 1, 1}));
            
            auto indices = _Concat({expandH, expandW}, -1);
            auto updates = _Transpose(outputDiff, {2, 3, 0, 1}); // HWNC
            std::vector<int> inputShape = {inH, inW, inB, inC};
            auto shape = _Const(inputShape.data(), {4}, NCHW, halide_type_of<int>());
            auto temp = _ScatterNd(indices, updates, shape, 0); // 0 for add
            res[0] = _Transpose(temp, {2, 3, 0, 1}); // NCHW
        }

        if (resizeType == 2) {
            const float scaleH = info.heightScale;
            const float scaleW = info.widthScale;
            const float offsetH = info.heightOffset;
            const float offsetW = info.widthOffset;

            vector<int> vecH, vecW;
            for (int i = 0; i < outH; i++) {
                vecH.push_back(i);
            }
            for (int i = 0; i < outW; i++) {
                vecW.push_back(i);
            }

            auto varH = _Const((void*)vecH.data(), {outH, 1, 1}, NCHW, halide_type_of<int>());
            auto varW = _Const((void*)vecW.data(), {1, outW, 1}, NCHW, halide_type_of<int>());

            // shape: outH * 1 * 1
            auto realH = _Cast<float>(varH) * _Scalar<float>(scaleH) + _Scalar<float>(offsetH);
            auto topH = _Cast<int>(_Floor(realH));
            auto h0 = Express::_Maximum(topH, _Scalar<int>(0));
            h0 = Express::_Minimum(h0, _Scalar<int>(inH-1));
            auto h1 = Express::_Maximum(topH+_Scalar<int>(1), _Scalar<int>(0));
            h1 = Express::_Minimum(h1, _Scalar<int>(inH-1));
            auto factorH = realH - _Cast<float>(topH);

            // shape: 1 * outW * 1
            auto realW = _Cast<float>(varW) * _Scalar<float>(scaleW) + _Scalar<float>(offsetW);
            auto leftW = _Cast<int>(_Floor(realW));
            auto w0 = Express::_Maximum(leftW, _Scalar<int>(0));
            w0 = Express::_Minimum(w0, _Scalar<int>(inW-1));
            auto w1 = Express::_Maximum(leftW+_Scalar<int>(1), _Scalar<int>(0));
            w1 = Express::_Minimum(w1, _Scalar<int>(inW-1));
            auto factorW = realW - _Cast<float>(leftW);

            auto one = _Scalar<float>(1.0f);
            // shape: outH * outW * 1
            auto f0 = (one - factorH) * (one - factorW);
            auto f1 = (one - factorH) * factorW;
            auto f2 = factorH * (one - factorW);
            auto f3 = factorH * factorW;
            
            // shape: outH * outW * 1
            auto expandH0 = h0 * _Cast<int>(_Const(1.0f, {1, outW, 1}));
            auto expandH1 = h1 * _Cast<int>(_Const(1.0f, {1, outW, 1}));
            auto expandW0 = w0 * _Cast<int>(_Const(1.0f, {outH, 1, 1}));
            auto expandW1 = w1 * _Cast<int>(_Const(1.0f, {outH, 1, 1}));

            // shape: outH * outW * 2
            auto h0w0 = _Concat({expandH0, expandW0}, -1);
            auto h0w1 = _Concat({expandH0, expandW1}, -1);
            auto h1w0 = _Concat({expandH1, expandW0}, -1);
            auto h1w1 = _Concat({expandH1, expandW1}, -1);
            
            auto updates = _Transpose(outputDiff, {2, 3, 0, 1}); // HWNC

            vector<int> factorShape = {outH, outW, 1, 1};
            auto u00 = _Reshape(f0, factorShape) * updates;
            auto u01 = _Reshape(f1, factorShape) * updates;
            auto u10 = _Reshape(f2, factorShape) * updates;
            auto u11 = _Reshape(f3, factorShape) * updates;

            std::vector<int> inputShape = {inH, inW, inB, inC};
            auto shape = _Const(inputShape.data(), {4}, NCHW, halide_type_of<int>());

            auto temp0 = _ScatterNd(h0w0, u00, shape, 0); // 0 for add
            auto temp1 = _ScatterNd(h0w1, u01, shape, 0);
            auto temp2 = _ScatterNd(h1w0, u10, shape, 0);
            auto temp3 = _ScatterNd(h1w1, u11, shape, 0);
            
            auto temp = temp0 + temp1 + temp2 + temp3;
            res[0] = _Transpose(temp, {2, 3, 0, 1}); // NCHW
        }

        if (resizeType == 3) {
            const float scaleH = info.heightScale;
            const float scaleW = info.widthScale;
            const float offsetH = info.heightOffset;
            const float offsetW = info.widthOffset;

            vector<int> vecH, vecW;
            for (int i = 0; i < outH; i++) {
                vecH.push_back(i);
            }
            for (int i = 0; i < outW; i++) {
                vecW.push_back(i);
            }

            auto varH = _Const((void*)vecH.data(), {outH, 1, 1}, NCHW, halide_type_of<int>());
            auto varW = _Const((void*)vecW.data(), {1, outW, 1}, NCHW, halide_type_of<int>());

            // shape: outH * 1 * 1
            auto realH = _Cast<float>(varH) * _Scalar<float>(scaleH) + _Scalar<float>(offsetH);
            auto topH = _Cast<int>(realH);
            auto h0 = Express::_Maximum(topH-_Scalar<int>(1), _Scalar<int>(0));
            h0 = Express::_Minimum(h0, _Scalar<int>(inH-1));
            auto h1 = Express::_Maximum(topH, _Scalar<int>(0));
            h1 = Express::_Minimum(h1, _Scalar<int>(inH-1));
            auto h2 = Express::_Maximum(topH+_Scalar<int>(1), _Scalar<int>(0));
            h2 = Express::_Minimum(h2, _Scalar<int>(inH-1));
            auto h3 = Express::_Maximum(topH+_Scalar<int>(2), _Scalar<int>(0));
            h3 = Express::_Minimum(h3, _Scalar<int>(inH-1));
            auto factorH = realH - _Floor(_Cast<float>(realH));

            // shape: 1 * outW * 1
            auto realW = _Cast<float>(varW) * _Scalar<float>(scaleW) + _Scalar<float>(offsetW);
            auto leftW = _Cast<int>(realW);
            auto w0 = Express::_Maximum(leftW-_Scalar<int>(1), _Scalar<int>(0));
            w0 = Express::_Minimum(w0, _Scalar<int>(inW-1));
            auto w1 = Express::_Maximum(leftW, _Scalar<int>(0));
            w1 = Express::_Minimum(w1, _Scalar<int>(inW-1));
            auto w2 = Express::_Maximum(leftW+_Scalar<int>(1), _Scalar<int>(0));
            w2 = Express::_Minimum(w2, _Scalar<int>(inW-1));
            auto w3 = Express::_Maximum(leftW+_Scalar<int>(2), _Scalar<int>(0));
            w3 = Express::_Minimum(w3, _Scalar<int>(inW-1));
            auto factorW = realW - _Floor(_Cast<float>(realW));

            auto sevenFive = _Scalar<float>(0.75f);
            auto one = _Scalar<float>(1.0f);
            auto oneTwoFive = _Scalar<float>(1.25f);
            auto two = _Scalar<float>(2.0f);
            auto twoTwoFive = _Scalar<float>(2.25f);
            auto three = _Scalar<float>(3.0f);
            auto five = _Scalar<float>(5.0f);
            auto six = _Scalar<float>(6.0f);
            
            VARPS fs = {sevenFive, one, oneTwoFive, two, twoTwoFive, three, five, six};
            // shape: outH * 1 * 4 * 1
            auto hFactors = CubicInterpolation2(_Reshape(factorH, {outH, 1, 1, 1}), fs, 2);
            // shape: 1 * outW * 1 * 4
            auto wFactors = CubicInterpolation2(_Reshape(factorW, {1, outW, 1, 1}), fs, 3);

            // shape: outH * outW * 1
            auto expandH0 = h0 * _Cast<int>(_Const(1.0f, {1, outW, 1}));
            auto expandH1 = h1 * _Cast<int>(_Const(1.0f, {1, outW, 1}));
            auto expandH2 = h2 * _Cast<int>(_Const(1.0f, {1, outW, 1}));
            auto expandH3 = h3 * _Cast<int>(_Const(1.0f, {1, outW, 1}));
            auto expandW0 = w0 * _Cast<int>(_Const(1.0f, {outH, 1, 1}));
            auto expandW1 = w1 * _Cast<int>(_Const(1.0f, {outH, 1, 1}));
            auto expandW2 = w2 * _Cast<int>(_Const(1.0f, {outH, 1, 1}));
            auto expandW3 = w3 * _Cast<int>(_Const(1.0f, {outH, 1, 1}));

            VARPS hIndices = {expandH0, expandH1, expandH2, expandH3};
            VARPS wIndices = {expandW0, expandW1, expandW2, expandW3};

            auto updates = _Transpose(outputDiff, {2, 3, 0, 1}); // HWNC
            std::vector<int> inputShape = {inH, inW, inB, inC};
            auto shape = _Const(inputShape.data(), {4}, NCHW, halide_type_of<int>());

            // shape: outH * outW * 2
            VARPS hwIndices;
            // shape: outH * outW * outB * outC
            VARPS hwUpdates;
            VARP tempRes = _Const(0.0f, inputShape, NCHW);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    // shape: outH * outW * 2
                    auto hwIndices = _Concat({hIndices[i], wIndices[j]}, -1);
                    // shape: outH * outW * outB * outC
                    auto hwUpdates = hFactors[i] * wFactors[j] * updates;
                    auto temp = _ScatterNd(hwIndices, hwUpdates, shape, 0); // 0 for add
                    tempRes = tempRes + temp;
                }
            }
            res[0] = _Transpose(tempRes, {2, 3, 0, 1}); // NCHW
        }

        res[0] = _Convert(res[0], inputInfo->order);
        return res;
    }
};

static const auto gRegister = []() {
    static InterpGrad _c;
    OpGrad::insert((int)OpType_Interp, &_c);
    OpGrad::insert((int)OpType_Resize, &_c);
    return true;
}();
