//
//  TorchTranspsoe.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

static VARP _IRange(VARP start, VARP limit, VARP delta) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_Range;
    auto rangeParam = new RangeT;
    rangeParam->Tidx = DataType_DT_INT32;
    op->main.type = OpParameter_Range;
    op->main.value = rangeParam;
    return Variable::create(Expr::create(std::move(op), {start, limit, delta}));
}

class TorchTranspsoeTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        std::unique_ptr<OpT> transpose(new OpT);
        transpose->type       = OpType_Transpose;
        transpose->main.type  = OpParameter_Transpose;
        transpose->main.value = new TransposeT;
        auto input = inputs[0];
        auto dim0 = inputs[1];
        auto dim1 = inputs[2];
        auto rank = _Rank(input);
        // dim0 = dim0 + rank * (dim0 < 0)
        auto realDim0 = _Add(dim0, _Multiply(rank, _Less(dim0, _Scalar(0))));
        // dim1 = dim1 + rank * (dim1 < 0)
        auto realDim1 = _Add(dim1, _Multiply(rank, _Less(dim1, _Scalar(0))));
        // minDim = min(dim0, dim1)
        auto minDim = _Minimum(realDim0, realDim1);
        // maxDim = max(dim0, dim1)
        auto maxDim = _Maximum(realDim0, realDim1);
        // left = [0, ..., minDim)
        auto left = _IRange(_Scalar(0), minDim, _Scalar(1));
        // middle = [minDim + 1, ..., maxDim)
        auto middle = _IRange(_Add(minDim, _Scalar(1)), maxDim, _Scalar(1));
        // right = [maxDim + 1, ..., rank)
        auto right = _IRange(_Add(maxDim, _Scalar(1)), rank, _Scalar(1));
        // perm = [left, maxDim, middle, minDim, right]
        auto perm = _Concat({left, maxDim, middle, minDim, right}, 0);
        std::vector<VARP> newInputs{inputs[0], perm};
        auto res = Expr::create(transpose.get(), newInputs);
        res->setName(opName);
        return res;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("transpose", std::shared_ptr<TorchExtraManager::Transform>(new TorchTranspsoeTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
