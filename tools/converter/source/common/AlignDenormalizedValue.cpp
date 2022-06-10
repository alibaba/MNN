//
//  AlignDenormalizedValue.cpp
//  MNNConverter
//
//  Created by MNN on 2022/01/07.
//  Copyright © 2018 - 2022 , Alibaba Group Holding Limited
//
#include "math.h"
#include "CommonUtils.hpp"
using namespace MNN;
static bool gPrinted = false;

void AlignDenormalizedValue(std::unique_ptr<MNN::OpT>& op) {

    const auto opType = op->main.type;
    switch (opType) {
        case MNN::OpParameter_Convolution2D: {
            auto param = op->main.AsConvolution2D();
            if (param->weight.empty()) {
                return;
            }
            auto weightPtr = param->weight.data();
            auto weightLastPtr = weightPtr + param->weight.size();
            bool aligned = false;
            float ValueMin = std::numeric_limits<float>().min();

            for (; weightPtr < weightLastPtr; ++weightPtr) { // has been speed up by auto vectorize
                aligned |= (*weightPtr) != 0 && fabs(*weightPtr) < ValueMin;
                *weightPtr = fabs(*weightPtr) < ValueMin ? 0 : (*weightPtr);
            }

            if (aligned) {
                if (!gPrinted) {
                    MNN_PRINT("caution: some weight absolute value is smaller than float min:%e, please check your training process.\n", ValueMin);
                    gPrinted = true;
                }
            }

            break;
        }
        default:
            break;
    }

};

void AlignDenormalizedValue(std::unique_ptr<MNN::NetT>& netT) {
    for (auto& op : netT->oplists) {
        AlignDenormalizedValue(op);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            AlignDenormalizedValue(op);
        }
    }
}

