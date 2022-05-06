//
//  TorchPad.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class TorchPadTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        auto info = op->main_as_Extra();
        PadValueMode mode = CONSTANT;
        if (nullptr != info->attr()) {
            for (int i = 0; i < info->attr()->size(); ++i) {
                const auto attr          = info->attr()->GetAs<Attribute>(i);
                const auto attributeName = attr->key()->str();
                if (attributeName == "mode") {
                    const std::map<std::string, PadValueMode> padValueModeMap = {{"constant", CONSTANT},
                                                                                 {"reflect", REFLECT}};
                    auto modeStr                                              = attr->s()->str();
                    if (padValueModeMap.find(modeStr) == padValueModeMap.end()) {
                        LOG(ERROR) << "MNN only support ['constant', 'reflect'] Pad mode";
                        return nullptr;
                    }
                    mode = padValueModeMap.at(modeStr);
                }
            }
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
            default:
                pad->main.AsPadParam()->mode = MNN::PadValueMode_CONSTANT;
                break;
        }
        // [N, C, H, W] -> [pad_W, W_pad, pad_H, H_pad]
        // [pad_W, W_pad, pad_H, H_pad] -> [pad_H, H_pad, pad_W, W_pad]
        // auto padsVar = _Reshape(_Transpose(_Reshape(inputs[1], {-1, 2}), {1, 0}), {-1});
        auto size = _Size(inputs[1]);
        auto dim = size / _Scalar(2);
        auto dims = _Stack({dim, dim});
        auto padsVar = _Reshape(_ReverseSequence(_Reshape(inputs[1], {-1, 2}), dims, 1, 0), {-1});
        // [pad_H, H_pad, pad_W, W_pad] -> [pad_N, N_pad, pad_C, C_pad, pad_H, H_pad, pad_W, W_pad]
        auto padPads = _Stack({_Rank(inputs[0]) * _Scalar(2) - size, _Scalar(0)});
        padsVar = _Pad(padsVar, padPads);
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
    TorchExtraManager::get()->insert("pad", std::shared_ptr<TorchExtraManager::Transform>(new TorchPadTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
