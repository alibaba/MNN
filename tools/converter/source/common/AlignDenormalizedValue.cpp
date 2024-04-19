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
                if (fabs(*weightPtr) < ValueMin) {           // To be compatible with lower gcc version than 5, should not use ternary expression along with value less than FLOAT_MIN.
                    *weightPtr = 0;
                }
            }

            if (aligned) {
                if (!gPrinted) {
                    MNN_PRINT("caution: some weight absolute values are not zero and smaller than float min:%e, please check your training process. op name:%s\n", ValueMin, op->name.c_str());
                    gPrinted = true;
                }
            }

            break;
        }
        default:
            break;
    }

};
