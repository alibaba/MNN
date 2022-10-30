
//
//  MatrixBandPartGrad.cpp
//  MNN
//
//  Created by MNN on 2022/08/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class MatrixBandPartGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        const auto& inputs = expr->inputs();
        auto input = inputs[0];
        auto lower = inputs[1]->readMap<int>()[0];
        auto upper = inputs[2]->readMap<int>()[0];
        std::vector<VARP> res(inputs.size());

        auto inputInfo = input->getInfo();
        int dimSize = inputInfo->dim.size();
        auto height = inputInfo->dim[dimSize-2];
        auto width = inputInfo->dim[dimSize-1];

        std::vector<float> mask;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                bool valid = (lower < 0 || (y - x) <= lower) && (upper < 0 || (x - y) <= upper);
                mask.emplace_back(valid ? 1.0f : 0.0f);
            }
        }

        auto maskVar = _Const(mask.data(), {height, width}, NCHW);
        res[0] = backwardOutput[0] * maskVar;

        return res;
    }
};

static const auto gRegister = []() {
    static MatrixBandPartGrad _c;
    OpGrad::insert((int)OpType_MatrixBandPart, &_c);
    return true;
}();
