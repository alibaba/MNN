//
//  RasterGrad.cpp
//  MNN
//
//  Created by MNN on 2022/10/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class RasterGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        std::vector<VARP> result(inputs.size(), nullptr);
        auto rasterInfo = expr->get()->main_as_Extra();
        const int32_t* regionData = nullptr;
        const int REGION_LENGTH = 11;
        for (int i=0; i<rasterInfo->attr()->size(); ++i) {
            auto attr = rasterInfo->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "region") {
                regionData = attr->list()->i()->data();
                MNN_ASSERT(inputs.size() * REGION_LENGTH == attr->list()->i()->size());
                break;
            }
        }
        for (int i=0; i<inputs.size(); ++i) {
            auto regionInfo = regionData + REGION_LENGTH * i;
            auto info = inputs[i]->getInfo();
            auto shape = info->dim;
            std::vector<int> curRegion(REGION_LENGTH);
            ::memcpy(curRegion.data(), regionInfo, REGION_LENGTH * sizeof(int));
            ::memcpy(curRegion.data(), regionInfo + 4, 4 * sizeof(int));
            ::memcpy(curRegion.data() + 4, regionInfo, 4 * sizeof(int));
            auto grad = _RasterRaw({backwardOutput[0]}, curRegion, shape, info->type, info->order);
            result[i] = grad;
        }
        return result;
    }
};

static const auto gRegister = []() {
    static RasterGrad _c;
    OpGrad::insert(OpType_Raster, &_c);
    return true;
}();
