//
//  RoiPoolGrad.cpp
//  MNN
//
//  Created by MNN on 2022/11/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class RoiPoolGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> res(1, nullptr);
        auto input = expr->inputs()[0];
        auto roi = expr->inputs()[1];

        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        auto param = forwardOp->main.AsRoiParameters();
        int pooledHeight = param->pooledHeight;
        int pooledWidth = param->pooledWidth;
        auto spatialScale = param->spatialScale;

        res[0] = _ROIPooling(input, roi, pooledHeight, pooledWidth, spatialScale, true, _Convert(backwardOutput[0], NC4HW4));
        res[0] = _Convert(res[0], input->getInfo()->order);

        return res;
    }
};

static const auto gRegister = []() {
    static RoiPoolGrad _c;
    OpGrad::insert(OpType_ROIPooling, &_c);
    return true;
}();
