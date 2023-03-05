//
//  OnnxTopK.cpp
//  MNNConverter
//
//  Created by MNN on 2020/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <numeric>

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxTopKTransformer : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);

        // values, indices = TopK(x, k, axis)
        // Select the top `k` valudes and indices from `x` at axis `axis`.
        //
        // Since tensorflow's TopK only performs at the last axis, so if the `axis` is
        // not the last axis, we had to transpose `x`, do topk with the transposed `x`,
        // and finally transpose the result back.
        auto inputs = expr->inputs();
        VARP kVar = (inputs.size() == 2 ? inputs[1] : nullptr);
        // Default Value See:
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK
        
        int axis = -1;
        bool largest = true;
        bool sorted = true;
        auto attrs = op->main_as_Extra()->attr();
        if (nullptr != attrs) {
            for (int i=0; i<attrs->size(); ++i) {
                auto attr = attrs->GetAs<Attribute>(i);
                if (attr->key()->str() == "axis") {
                    axis = attr->i();
                } else if (attr->key()->str() == "k" && inputs.size() == 1) {
                    kVar = _Scalar<int32_t>(attr->i());
                } else if (attr->key()->str() == "largest") {
                    largest = attr->i();
                }
            }
        }

        std::unique_ptr<TopKV2T> onnxTopKParam(new TopKV2T);
        onnxTopKParam->T      = DataType_DT_FLOAT;
        onnxTopKParam->sorted = sorted;
        onnxTopKParam->largest = largest;

        std::unique_ptr<OpT> onnxTopKOp(new OpT);
        onnxTopKOp->name       = op->name()->str();
        onnxTopKOp->type       = OpType_TopKV2;
        onnxTopKOp->main.type  = OpParameter_TopKV2;
        onnxTopKOp->main.value = onnxTopKParam.release();

        if (axis < 0 && inputs[0]->getInfo()) {
            axis += inputs[0]->getInfo()->dim.size();
        }
        EXPRP output = Expr::create(onnxTopKOp.get(), {inputs[0], kVar, _Scalar<int>(axis)}, 2);
        output->setName(expr->name());
        Variable::create(output, 0)->setName(expr->outputName(0));
        Variable::create(output, 1)->setName(expr->outputName(1));
        return output;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("TopK", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxTopKTransformer));
    return true;
}();

} // namespace Express
} // namespace MNN
