//
//  ReluGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReluGrad.hpp"
#include "core/Macro.h"
#include <string.h>
using namespace std;
using namespace MNN;
using namespace MNN::Express;
class PReluGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(1, nullptr);
        auto op = expr->get();
        auto input = expr->inputs()[0];
        auto mask = _Relu(_Sign(input));
        auto prelu = op->main_as_PRelu();
        if (prelu->slope()->size() == 1) {
            auto slope = prelu->slope()->data()[0];
            result[0] = (mask + (_Scalar<float>(1.0f) - mask) * _Scalar<float>(slope)) * backwardOutput[0];
            return result;
        }
        auto channel = prelu->slope()->size();
        std::vector<float> scale(channel);
        ::memcpy(scale.data(), prelu->slope()->data(), channel * sizeof(float));
        std::vector<float> bias(channel, 0.0f);
        auto outputSecond = _Scale(backwardOutput[0], channel, std::move(scale), std::move(bias));
        result[0] = mask * backwardOutput[0] + (_Scalar<float>(1.0f) - mask) * outputSecond;
//        auto diffInfo = result[0]->getInfo();
//        auto inputInfo = input->getInfo();
//        for (int i=0; i<diffInfo->dim.size(); ++i) {
//            MNN_ASSERT(diffInfo->dim[i] == inputInfo->dim[i]);
//            MNN_PRINT("%s, %d, %d - %d\n", expr->name().c_str(), i, diffInfo->dim[i], inputInfo->dim[i]);
//        }
//        MNN_ASSERT(diffInfo->order == inputInfo->order);
        return result;
    }

};
class ReluGrad : public OpGrad {
public:
    ReluGrad() {
        mType = SEMI_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(1, nullptr);
        auto op = expr->get();
        auto input = expr->inputs()[0];
        auto mask = _Relu(_Sign(input));
        if (nullptr != op->main_as_Relu() && op->main_as_Relu()->slope() != 0.0f) {
            auto mask2 = _Cast<float>(_Less(input, _Scalar(0.0f)));
            result[0] = (mask + mask2 * _Scalar<float>(op->main_as_Relu()->slope())) * backwardOutput[0];
            return result;
        }
        result[0] = mask * backwardOutput[0];
        return result;
    }
};
class Relu6Grad : public OpGrad {
public:
    Relu6Grad() {
        mType = SEMI_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result{nullptr};
        auto op = expr->get();
        MNN_ASSERT(nullptr != op);
        auto relu6 = op->main_as_Relu6();
        MNN_ASSERT(nullptr != relu6);
        auto input = expr->inputs()[0];
        auto mask0 = _Cast<float>(_Greater(input, _Scalar(relu6->minValue())));
        auto mask1 = _Cast<float>(_Less(input, _Scalar(relu6->maxValue())));

        result[0] = mask0 * mask1 * backwardOutput[0];
        return result;
    }
};
static const auto gRegister = []() {
    static ReluGrad _c;
    OpGrad::insert(OpType_ReLU, &_c);
    static Relu6Grad _d;
    OpGrad::insert(OpType_ReLU6, &_d);
    static PReluGrad _e;
    OpGrad::insert(OpType_PReLU, &_e);
    return true;
}();
