//
//  RasterGrad.cpp
//  MNN
//
//  Created by MNN on 2022/10/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/TensorUtils.hpp"
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
        std::vector<int32_t> regionDataHolder;
        if (nullptr != rasterInfo && nullptr != rasterInfo->attr()) {
            for (int i=0; i<rasterInfo->attr()->size(); ++i) {
                auto attr = rasterInfo->attr()->GetAs<Attribute>(i);
                if (attr->key()->str() == "region") {
                    regionData = attr->list()->i()->data();
                    MNN_ASSERT(inputs.size() * REGION_LENGTH == attr->list()->i()->size());
                    break;
                }
            }
        } else {
            regionDataHolder.resize(inputs.size() * REGION_LENGTH);
            regionData = regionDataHolder.data();
            auto outputTensor = Variable::create(expr)->getTensor();
            auto des = TensorUtils::getDescribe(outputTensor);
            MNN_ASSERT(des->regions.size() == inputs.size());
            for (int i=0; i<inputs.size(); ++i) {
                auto& r = des->regions[i];
                auto dstPtr = regionDataHolder.data() + REGION_LENGTH * i;
                dstPtr[0] = r.src.offset;
                dstPtr[1] = r.src.stride[0];
                dstPtr[2] = r.src.stride[1];
                dstPtr[3] = r.src.stride[2];

                dstPtr[4] = r.dst.offset;
                dstPtr[5] = r.dst.stride[0];
                dstPtr[6] = r.dst.stride[1];
                dstPtr[7] = r.dst.stride[2];
                
                dstPtr[8] = r.size[0];
                dstPtr[9] = r.size[1];
                dstPtr[10] = r.size[2];
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
