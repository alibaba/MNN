//
//  WinogradInt8Helper.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WinogradInt8Helper_hpp
#define WinogradInt8Helper_hpp

#include <vector>
#include "core/Macro.h"

namespace MNN {
class MNN_PUBLIC WinogradInt8Helper {
public:
    static void transformWeight(const std::vector<float>& weight, std::vector<float>& transWeight,
                                std::vector<int>& attrs, int oc, int ic, int kernelY, int kernelX) {
        bool conv2d = (kernelY != 1 && kernelX != 1);
        const int ALPHA = 4;
        using Vec = std::vector<std::pair<int, int>>;
        auto partitionFunc = [=](int kernel) -> Vec {
            Vec partition;
            for (int i = 0, count; i < kernel; i += count) {
                if (i + 4 == kernel) {
                    count = 2;
                } else {
                    count = ALIMIN(kernel - i, 3);
                }
                partition.emplace_back(i, count);
            }
            return partition;
        };
        auto transWeightFunc = [](const float* weight, float* transWeight, int step, int kernel) {
            if (kernel == 3) {
                transWeight[0 * step] = weight[0];
                transWeight[1 * step] = 0.5 * (weight[0] + weight[1] + weight[2]);
                transWeight[2 * step] = 0.5 * (weight[0] - weight[1] + weight[2]);
                transWeight[3 * step] = weight[2];
            } else if (kernel == 2) {
                transWeight[0 * step] = weight[0];
                transWeight[1 * step] = 0.5 * (weight[0] + weight[1]);
                transWeight[2 * step] = 0.5 * (weight[0] - weight[1]);
                transWeight[3 * step] = weight[1];
            }
        };
        auto yAttrs = partitionFunc(kernelY), xAttrs = partitionFunc(kernelX);
        attrs.push_back(kernelY);
        attrs.push_back(kernelX);
        attrs.push_back(kernelY == 1 ? 1 : yAttrs.size() * ALPHA);
        attrs.push_back(kernelX == 1 ? 1 : xAttrs.size() * ALPHA);
        int transKernel = attrs[2] * attrs[3];
        transWeight.resize(oc * ic * transKernel);
        float* dstOrigin = transWeight.data();
        for (auto& yAttr : yAttrs) {
            for (auto& xAttr : xAttrs) {
                int ky = yAttr.second, kx = xAttr.second, alphaX = (ky == 1 ? 1 : ALPHA);
                auto srcOrigin = weight.data() + yAttr.first * kernelX + xAttr.first;
                for (int i = 0; i < oc * ic; ++i) {
                    const float* originData = srcOrigin + i * kernelY * kernelX;
                    auto data = dstOrigin + i * transKernel;
                    if (conv2d) {
                        std::vector<float> tmp(transKernel);
                        for (int ky_ = 0; ky_ < ky; ++ky_) {
                            transWeightFunc(originData + ky_ * kernelX, tmp.data() + ky_, ky, kx);
                        }
                        for (int kx_ = 0; kx_ < alphaX; ++kx_) {
                            transWeightFunc(tmp.data() + kx_ * ky, data + kx_, alphaX, ky);
                        }
                    } else {
                        transWeightFunc(originData, data, 1, ALIMAX(ky, kx));
                    }
                }
                dstOrigin += ALPHA * (conv2d ? ALPHA : 1);
                attrs.push_back(yAttr.first);
                attrs.push_back(xAttr.first);
                attrs.push_back(ky == 1 ? 1 : (ALPHA - ky + 1));
                attrs.push_back(kx == 1 ? 1 : (ALPHA - kx + 1));
            }
        }
    }
};
}

#endif // WinogradInt8Helper_hpp
