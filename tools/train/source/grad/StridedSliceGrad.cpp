//
//  StridedSliceGrad.cpp
//  MNN
//
//  Created by MNN on 2022/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class StridedSliceGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        std::vector<VARP> res(inputs.size(), nullptr);
        auto outputDiff = backwardOutput[0];

        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        int beginMask = forwardOp->main.AsStridedSliceParam()->beginMask;
        int endMask = forwardOp->main.AsStridedSliceParam()->endMask;
        int ellipsisMask = forwardOp->main.AsStridedSliceParam()->ellipsisMask;
        int newAxisMask = forwardOp->main.AsStridedSliceParam()->newAxisMask;
        int shrinkAxisMask = forwardOp->main.AsStridedSliceParam()->shrinkAxisMask;

        auto input = inputs[0];
        auto begin = inputs[1];
        auto end = inputs[2];
        auto stride = inputs[3];

        auto zeros = _ZerosLike(input);
        res[0] = _StridedSliceWrite(zeros, begin, end, stride, outputDiff, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);

        return res;
    }
};

static const auto gRegister = []() {
    static StridedSliceGrad _c;
    OpGrad::insert(OpType_StridedSlice, &_c);
    return true;
}();
