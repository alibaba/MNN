//
//  OnnxMean.cpp
//  MNNConverter
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxMeanTransform : public OnnxExtraManager::Transform {
public:
    EXPRP onExecute(EXPRP expr) const override {
        const auto& inputs = expr->inputs();
        if (inputs.empty()) {
            return nullptr;
        }
        if (inputs.size() == 1) {
            OpT identity;
            identity.type = OpType_Identity;
            auto result = Expr::create(&identity, inputs);
            result->setName(expr->name());
            return result;
        }

        // Binary addition broadcasts, while packing requires identical input shapes.
        auto scale = _Scalar<float>(1.0f / static_cast<float>(inputs.size()));
        auto result = _Multiply(inputs[0], scale);
        for (size_t i = 1; i < inputs.size(); ++i) {
            result = _Add(result, _Multiply(inputs[i], scale));
        }
        result->setName(expr->name());
        return result->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Mean", std::make_shared<OnnxMeanTransform>());
    return true;
}();

} // namespace Express
} // namespace MNN
