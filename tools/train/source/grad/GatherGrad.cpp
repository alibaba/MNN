//
//  GatherGrad.cpp
//  MNN
//
//  Created by MNN on 2021/02/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class GatherGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto op = expr->get();
        const auto& inputs = expr->inputs();
        auto param = inputs[0];
        auto indice = inputs[1];
        auto dims = indice->getInfo()->dim;
        dims.emplace_back(1);
        indice = _Reshape(indice, dims);
        int axis = 0;
        std::vector<VARP> res(inputs.size());
        if (inputs.size() > 2) {
            axis = inputs[2]->readMap<int>()[0];
        }
        if (axis != 0) {
            MNN_ERROR("Current's don't support axis != 0 grad for gather\n");
            return res;
        }
        auto shape = _Shape(param);
        auto diff = _Unsqueeze(backwardOutput[0], {axis});
        res[0] = _ScatterNd(indice, diff, shape);
        return res;
    }
};

static const auto gRegister = []() {
    static GatherGrad _c;
    OpGrad::insert((int)OpType_GatherV2, &_c);
    return true;
}();
