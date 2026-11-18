//
//  OnnxMod.cpp
//  MNNConverter
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxModTransform : public OnnxExtraManager::Transform {
public:
    EXPRP onExecute(EXPRP expr) const override {
        const auto& inputs = expr->inputs();
        if (inputs.size() != 2) {
            return nullptr;
        }
        bool fmod = false;
        if (const auto* attrs = expr->get()->main_as_Extra()->attr()) {
            for (const auto* attr : *attrs) {
                if (attr->key()->str() == "fmod") {
                    fmod = attr->i() != 0;
                }
            }
        }

        auto opType = BinaryOpOperation_MOD;
        if (fmod) {
            const auto* info = inputs[0]->getInfo();
            if (info == nullptr) {
                return nullptr;
            }
            // MOD uses fmod for floats; the integer truncating path uses FLOORMOD.
            if (info->type.code != halide_type_float) {
                opType = BinaryOpOperation_FLOORMOD;
            }
        }
        OpT op;
        op.type = OpType_BinaryOp;
        op.main.type = OpParameter_BinaryOp;
        auto* param = new BinaryOpT;
        param->opType = opType;
        op.main.value = param;
        auto result = Expr::create(&op, inputs);
        result->setName(expr->name());
        return result;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Mod", std::make_shared<OnnxModTransform>());
    return true;
}();

} // namespace Express
} // namespace MNN
