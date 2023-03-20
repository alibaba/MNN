//
//  RoiAlignGrad.cpp
//  MNN
//
//  Created by MNN on 2022/12/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class RoiAlignGrad : public OpGrad {
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
        int samplingRatio = param->samplingRatio;
        bool aligned = param->aligned;
        auto poolType = param->poolType;

        res[0] = _ROIAlign(input, roi, pooledHeight, pooledWidth, spatialScale, samplingRatio, aligned, PoolingMode(poolType), true, _Convert(backwardOutput[0], NC4HW4));
        res[0] = _Convert(res[0], input->getInfo()->order);

        return res;
    }
};

static const auto gRegister = []() {
    static RoiAlignGrad _c;
    OpGrad::insert(OpType_ROIAlign, &_c);
    return true;
}();
