//
//  OnnxPad.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include <string>
#include <vector>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class OnnxPadTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();

        PadValueMode mode = CONSTANT;
        VARP padsVar;
        bool padsFromInput = true;

        auto info = op->main_as_Extra();
        if (nullptr != info->attr()) {
            for (int i = 0; i < info->attr()->size(); ++i) {
                const auto attr          = info->attr()->GetAs<Attribute>(i);
                const auto attributeName = attr->key()->str();
                if (attributeName == "mode") {
                    const std::map<std::string, PadValueMode> padValueModeMap = {
                        {"constant", CONSTANT}, {"reflect", REFLECT}, {"edge", EDGE}
                    };
                    auto modeStr                                              = attr->s()->str();
                    if (padValueModeMap.find(modeStr) == padValueModeMap.end()) {
                        LOG(ERROR) << "MNN only support ['constant', 'reflect'] Pad mode";
                        return nullptr;
                    }
                    mode = padValueModeMap.at(modeStr);
                } else if (attributeName == "pads") {
                    padsFromInput = false;
                    auto padList  = attr->list()->i();
                    int size      = padList->size();
                    std::vector<int> pads(size);
                    for (int s = 0; s < size / 2; ++s) {
                        pads[s * 2]     = padList->Get(s);
                        pads[s * 2 + 1] = padList->Get(s + size / 2);
                    }
                    padsVar = _Const(pads.data(), {(int)pads.size()}, NCHW, halide_type_of<int>());
                }
            }
        }
        if (padsFromInput) {
            if (inputs.size() == 1) {
                LOG(ERROR) << "MNN need pad value in attr or other node";
                return nullptr;
            }
            /* Pytorch's Pad exported by Onnx (opset_version=11) have complicated subgraph.
               Pad values in input node is [before_pads, after_pads], which is not same as MNN pads order.
               Example: pad2d, onnx: [left, upper, right, bottom], MNN: [left, right, upper, bottom]
               So we need this order converting subgraph (all const, not affect inference speed).
             */
            padsVar = _Reshape(_Transpose(_Reshape(inputs[1], {2, -1}), {1, 0}), {-1});
        }
        std::unique_ptr<OpT> pad(new OpT);
        pad->type       = OpType_Padding;
        pad->main.type  = OpParameter_PadParam;
        pad->main.value = new PadParamT;
        switch (mode) {
            case CONSTANT:
                pad->main.AsPadParam()->mode = MNN::PadValueMode_CONSTANT;
                break;
            case SYMMETRIC:
                pad->main.AsPadParam()->mode = MNN::PadValueMode_SYMMETRIC;
                break;
            case REFLECT:
                pad->main.AsPadParam()->mode = MNN::PadValueMode_REFLECT;
                break;
            case EDGE:
                pad->main.AsPadParam()->mode = MNN::PadValueMode_EDGE;
                break;
            default:
                pad->main.AsPadParam()->mode = MNN::PadValueMode_CONSTANT;
                break;
        }
        std::vector<VARP> newInputs{inputs[0], padsVar};
        if (inputs.size() > 2) {
            newInputs.emplace_back(inputs[2]);
        }
        auto res = Expr::create(pad.get(), newInputs);
        res->setName(opName);
        return res;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Pad", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxPadTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
