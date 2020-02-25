//
//  TensorConvertGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TensorConvertGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class TensorConvertGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result{nullptr};
        auto originInput = expr->inputs()[0];
        result[0]        = _Convert(backwardOutput[0], originInput->getInfo()->order);
        return result;
    }
};
static const auto gRegister = []() {
    static TensorConvertGrad _c;
    OpGrad::insert(OpType_ConvertTensor, &_c);
    return true;
}();
