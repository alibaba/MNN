//
//  WinogradInt8Attr.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WinogradInt8Attr_hpp
#define WinogradInt8Attr_hpp

#include <cstdint>
#include <vector>
#include <MNN/expr/ExprCreator.hpp>
#include "core/Macro.h"

namespace MNN {
class WinogradInt8Attr {
public:
    struct Attr {
        int kyStart, kxStart, kernelY, kernelX, unitY, unitX;
        std::vector<float> inputScales, weightScales;
        std::vector<int32_t> inputZeroPoints;
    };
    void add(int kyStart, int kxStart, int kernelY, int kernelX, int unitY, int unitX,
             std::vector<float> inputScales = {}, std::vector<float> weightScales = {}, std::vector<int32_t> inputZeroPoints = {}) {
        Attr attr {kyStart, kxStart, kernelY, kernelX, unitY, unitX, inputScales, weightScales, inputZeroPoints};
        attrs.push_back(attr);
    }
    Express::VARP turnToWinogradConv(Express::VARP originConv) {
        if (attrs.size() == 0) {
            return originConv;
        }
        auto conv2d = originConv->expr().first->get()->main_as_Convolution2D();
        if (conv2d->bias() == nullptr || conv2d->bias()->size() == 0 || conv2d->quanParameter() == nullptr) {
            MNN_ERROR("Invalid origin conv op\n");
            return nullptr;
        }
        std::unique_ptr<MNN::OpT> op( originConv->expr().first->get()->UnPack());
        op->main.AsConvolution2D()->symmetricQuan->winogradAttr = encode();
        return (Express::Variable::create(Express::Expr::create(op.get(), originConv->expr().first->inputs())));
    }
    std::vector<Attr> attrs;
private:
    std::vector<int32_t> encode() {
        std::vector<int32_t> bin;
        bin.push_back(0);
        bin.push_back(attrs.size());
        auto insert_vec = [&](std::vector<int32_t>& bin, const void* data, int len) {
            auto pos = bin.size();
            bin.resize(pos + len);
            ::memcpy(bin.data() + pos, data, len * 4); // sizeof(float) = sizeof(int32_t) = 4
        };
        for (const auto& attr : attrs) {
            std::vector<int32_t> attr_bin;
            attr_bin.insert(attr_bin.begin(), {attr.kyStart, attr.kxStart, attr.kernelY, attr.kernelX, attr.unitY, attr.unitX});
            insert_vec(attr_bin, (const void*)attr.inputScales.data(), attr.inputScales.size());
            insert_vec(attr_bin, (const void*)attr.inputZeroPoints.data(), attr.inputZeroPoints.size());
            insert_vec(attr_bin, (const void*)attr.weightScales.data(), attr.weightScales.size());
            attr_bin.insert(attr_bin.begin(), attr_bin.size());
            bin.insert(bin.end(), attr_bin.begin(), attr_bin.end());
        }
        return bin;
    }
};
}

#endif // WinogradInt8Attr_hpp
