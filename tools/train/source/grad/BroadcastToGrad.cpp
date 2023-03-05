//
//  BroadcastToGrad.cpp
//  MNN
//
//  Created by MNN on 2022/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class BroadcastToGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        std::vector<VARP> res(inputs.size(), nullptr);
        auto outputDiff = backwardOutput[0];
        auto outputDiffInfo = outputDiff->getInfo();

        auto input = inputs[0];
        auto inputInfo = input->getInfo();

        std::vector<int> reduceDims;
        bool keepDim = true;

        if (inputInfo->dim.size() < outputDiffInfo->dim.size()) {
            // case like: shape(2, 3, 1) ==> shape(7, 2, 3, 3)
            // will only be handled a part here
            // because we need keepDim = false for dim[0] = 7
            // and keepDim = true for dim[-1] = 3
            auto diff = (int)outputDiffInfo->dim.size() - (int)inputInfo->dim.size();
            for (int i = 0; i < diff; ++i) {
                reduceDims.emplace_back(i);
            }
            keepDim = false;
        } else {
            for (int i = 0; i < outputDiffInfo->dim.size(); ++i) {
                if (outputDiffInfo->dim[i] > 1 && inputInfo->dim[i] == 1) {
                    reduceDims.emplace_back(i);
                }
            }
            keepDim = true;
        }

        if (!reduceDims.empty()) {
            res[0] = _ReduceSum(outputDiff, reduceDims, keepDim);
            // for case like: shape(2, 3, 1) ==> shape(7, 2, 3, 3)
            if (keepDim == false) {
                reduceDims.clear();
                auto diff = (int)outputDiffInfo->dim.size() - (int)inputInfo->dim.size();
                for (int j = 0; j < inputInfo->dim.size(); j++) {
                    if (outputDiffInfo->dim[j + diff] > 1 && inputInfo->dim[j] == 1) {
                        reduceDims.emplace_back(j);
                    }
                }
                keepDim = true;
                if (!reduceDims.empty()) {
                    res[0] = _ReduceSum(outputDiff, reduceDims, keepDim);
                }
            }
        } else {
            res[0] = outputDiff;
        }

        return res;
    }
};

static const auto gRegister = []() {
    static BroadcastToGrad _c;
    OpGrad::insert(OpType_BroadcastTo, &_c);
    return true;
}();
