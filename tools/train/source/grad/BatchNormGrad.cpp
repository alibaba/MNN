//
//  BatchNormGrad.cpp
//  MNN
//
//  Created by MNN on 2019/11/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BatchNormGrad.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <vector>
#include "core/Macro.h"

using namespace std;
using namespace MNN;
using namespace MNN::Express;

class BatchNormGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& outputs,
                                              const std::vector<Express::VARP>& backDiff) override {
        // input, scale, bias, running_mean, running_variance, epsilon, momentum, is_training
        // scale and bias are learnable
        std::shared_ptr<OpT> forwardOp(expr->get()->UnPack());
        std::vector<Express::VARP> res;
        auto inputs = expr->inputs();
        res.resize(inputs.size()); // only back propgate to input, scale, bias

        auto input          = inputs[0];
        auto scale          = inputs[1];
        auto bias           = inputs[2];
        auto output         = outputs[0];
        auto normalizedData = outputs[3]; // (input - sample_mean) / sqrt(sample_variance + epsilon)
        auto rSampleStd     = outputs[4]; // rsqrt(sample_variance + epsilon)

        MNN_ASSERT(scale->getInfo()->dim.size() == 1);
        // reshape in order to use broadcast
        auto factor = _Reshape(_Multiply(scale, rSampleStd), {1, scale->getInfo()->dim[0], 1, 1}, NCHW);
        res[0]      = _Multiply(backDiff[0], factor);
        res[0]->setName(forwardOp->name + "_BN_Input_Grad");

        res[1] = _ReduceSum(_Multiply(backDiff[0], normalizedData), {0, 2, 3}, false);
        res[1]->setName(forwardOp->name + "_BN_Scale_Grad");

        res[2] = _ReduceSum(backDiff[0], {0, 2, 3}, false);
        res[2]->setName(forwardOp->name + "_BN_Bias_Grad");

        return res;
    }
};
static const auto gRegister = []() {
    static BatchNormGrad _c;
    OpGrad::insert(OpType_BatchNorm, &_c);
    return true;
}();
