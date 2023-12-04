
//
//  GridSampleGrad.cpp
//  MNN
//
//  Created by MNN on 2022/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class GridSampleGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto op = expr->get();
        const auto& inputs = expr->inputs();
        std::vector<VARP> res(inputs.size());
#ifndef GRIDSAMPLER_GRAD_DONT_CREATENEWOP
        std::unique_ptr<MNN::OpT> gradOp(op->UnPack());
        gradOp->name.clear();
        gradOp->main.AsGridSample()->backward = true;
        auto gradForParameterExpr = Expr::create(gradOp.get(), {backwardOutput[0], inputs[1], _Shape(inputs[0], NCHW)});
        res[0] = Variable::create(gradForParameterExpr);
        return res;
#else
        auto param = op->main_as_GridSample();
        auto sampleMode = param->mode();
        auto padMode = param->paddingMode();
        auto alignCorners = param->alignCorners();

        auto input = inputs[0];
        auto grid = inputs[1];
        
        auto inputInfo = input->getInfo();
        MNN_ASSERT(nullptr != inputInfo && (inputInfo->dim.size() == 4 || inputInfo->dim.size() == 5));
        auto outputDiff = backwardOutput[0];
        outputDiff = _Convert(outputDiff, NCHW);
        auto diffInfo = outputDiff->getInfo();

        const int inB = inputInfo->dim[0];
        const int inC = inputInfo->dim[1];
        const int inH = inputInfo->dim[2];
        const int inW = inputInfo->dim[3];

        const int outB = diffInfo->dim[0];
        const int outC = diffInfo->dim[1];
        const int outH = diffInfo->dim[2];
        const int outW = diffInfo->dim[3];

        auto one = _Scalar<float>(1.0f);
        auto zeroFive = _Scalar<float>(0.5f);
        std::vector<float> inShape = {float(inW), float(inH)};
        auto inShapeVar = _Const((void*)inShape.data(), {1, 1, 1, 2}, NCHW);
        outputDiff = _Transpose(outputDiff, {0, 2, 3, 1}); // NHWC
        
        VARP sourceCord = nullptr;
        if (alignCorners) {
            sourceCord = (grid + one) * (inShapeVar - one) * zeroFive; // N, outH, outW, 2 (w, h)
        } else {
            sourceCord = ((grid + one) * inShapeVar - one) * zeroFive; // N, outH, outW, 2 (w, h)
        }

        if (sampleMode == SampleMode_NEAREST) {
            auto indices = _Cast<int32_t>(_Floor(sourceCord + zeroFive)); // N, outH, outW, 2 (w, h)

            { // handle three pad modes
                auto cords = _Split(indices, {1, 1}, -1);
                auto wCord = cords[0];
                auto hCord = cords[1];
                auto zero = _Scalar<int>(0);
                auto one = _Scalar<int>(1);
                auto wVar = _Scalar<int>(inW);
                auto hVar = _Scalar<int>(inH);
                if (padMode == BorderMode_ZEROS) {
                    auto wMask1 = one - _Less(wCord, zero);
                    auto wMask2 = one - _GreaterEqual(wCord, wVar);
                    auto hMask1 = one - _Less(hCord, zero);
                    auto hMask2 = one - _GreaterEqual(hCord, hVar);
                    auto mask = wMask1 * wMask2 * hMask1 * hMask2;
                    outputDiff = outputDiff * _Cast<float>(mask);
                }
                wCord = Express::_Maximum(wCord, zero);
                wCord = Express::_Minimum(wCord, wVar - one);
                hCord = Express::_Maximum(hCord, zero);
                hCord = Express::_Minimum(hCord, hVar - one);
                indices = _Concat({wCord, hCord}, -1);
            }

            std::vector<int> bIndices;
            for (int i = 0; i < outB; i++) {
                for (int j = 0; j < outH*outW; j++) {
                    bIndices.push_back(i);
                }
            }
            auto bIndicesVar = _Const((void*)bIndices.data(), {outB, outH, outW, 1}, NCHW, halide_type_of<int>());
            
            indices = _Concat({bIndicesVar, indices}, -1);
            auto updates = outputDiff;
            std::vector<int> inputShape = {inB, inW, inH, inC};
            auto shape = _Const(inputShape.data(), {4}, NCHW, halide_type_of<int>());
            auto temp = _ScatterNd(indices, updates, shape, 0); // 0 for add
            res[0] = _Transpose(temp, {0, 3, 2, 1}); // NCHW
        } else if (sampleMode == SampleMode_BILINEAR) {
            auto w0h0 = _Cast<int>(_Floor(sourceCord));
            auto w1h1 = _Cast<int>(_Ceil(sourceCord));
            auto factors0 = sourceCord - _Cast<float>(w0h0);

            auto cords0 = _Split(w0h0, {1, 1}, -1);
            auto w0 = cords0[0];
            auto h0 = cords0[1];
            auto cords1 = _Split(w1h1, {1, 1}, -1);
            auto w1 = cords1[0];
            auto h1 = cords1[1];

            auto fs0 = _Split(factors0, {1, 1}, -1);
            auto wf0 = fs0[0];
            auto hf0 = fs0[1];
            auto wf1 = _Scalar<float>(1.0f) - wf0;
            auto hf1 = _Scalar<float>(1.0f) - hf0;

            auto w00 = wf1 * hf1;
            auto w01 = wf1 * hf0;
            auto w10 = wf0 * hf1;
            auto w11 = wf0 * hf0;

            auto updates = outputDiff;
            auto u00 = updates * w00;
            auto u01 = updates * w01;
            auto u10 = updates * w10;
            auto u11 = updates * w11;

            { // handle three pad modes
                auto zero = _Scalar<int>(0);
                auto one = _Scalar<int>(1);
                auto wVar = _Scalar<int>(inW);
                auto hVar = _Scalar<int>(inH);
                if (padMode == BorderMode_ZEROS) {
                    auto wMask0 = one - _Less(w0, zero);
                    auto wMask1 = one - _GreaterEqual(w0, wVar);
                    auto wMask2 = one - _Less(w1, zero);
                    auto wMask3 = one - _GreaterEqual(w1, wVar);
                    auto hMask0 = one - _Less(h0, zero);
                    auto hMask1 = one - _GreaterEqual(h0, hVar);
                    auto hMask2 = one - _Less(h1, zero);
                    auto hMask3 = one - _GreaterEqual(h1, hVar);

                    auto mask00 = wMask0 * wMask1 * hMask0 * hMask1;
                    auto mask01 = wMask0 * wMask1 * hMask2 * hMask3;
                    auto mask10 = wMask2 * wMask3 * hMask0 * hMask1;
                    auto mask11 = wMask2 * wMask3 * hMask2 * hMask3;

                    u00 = u00 * _Cast<float>(mask00);
                    u01 = u01 * _Cast<float>(mask01);
                    u10 = u10 * _Cast<float>(mask10);
                    u11 = u11 * _Cast<float>(mask11);
                }
                w0 = Express::_Maximum(w0, zero);
                w0 = Express::_Minimum(w0, wVar - one);
                w1 = Express::_Maximum(w1, zero);
                w1 = Express::_Minimum(w1, wVar - one);
                h0 = Express::_Maximum(h0, zero);
                h0 = Express::_Minimum(h0, hVar - one);
                h1 = Express::_Maximum(h1, zero);
                h1 = Express::_Minimum(h1, hVar - one);
            }

            std::vector<int> bIndices;
            for (int i = 0; i < outB; i++) {
                for (int j = 0; j < outH*outW; j++) {
                    bIndices.push_back(i);
                }
            }
            auto bIndicesVar = _Const((void*)bIndices.data(), {outB, outH, outW, 1}, NCHW, halide_type_of<int>());

            auto bw0h1 = _Concat({bIndicesVar, w0, h1}, -1);
            auto bw1h0 = _Concat({bIndicesVar, w1, h0}, -1);
            auto bw0h0 = _Concat({bIndicesVar, w0, h0}, -1);
            auto bw1h1 = _Concat({bIndicesVar, w1, h1}, -1);

            std::vector<int> inputShape = {inB, inW, inH, inC};
            auto shape = _Const(inputShape.data(), {4}, NCHW, halide_type_of<int>());
            auto temp0 = _ScatterNd(bw0h0, u00, shape, 0); // 0 for add
            auto temp1 = _ScatterNd(bw0h1, u01, shape, 0);
            auto temp2 = _ScatterNd(bw1h0, u10, shape, 0);
            auto temp3 = _ScatterNd(bw1h1, u11, shape, 0);

            auto temp = temp0 + temp1 + temp2 + temp3;
            res[0] = _Transpose(temp, {0, 3, 2, 1}); // NCHW
        }

        res[0] = _Convert(res[0], inputInfo->order);
        return res;
#endif
    }
};

static const auto gRegister = []() {
    static GridSampleGrad _c;
    OpGrad::insert((int)OpType_GridSample, &_c);
    OpGrad::insert((int)OpType_Texture, &_c);
    return true;
}();
