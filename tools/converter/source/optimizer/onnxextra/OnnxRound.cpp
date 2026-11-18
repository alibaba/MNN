//
//  OnnxRound.cpp
//  MNNConverter
//

#include <MNN/expr/ExprCreator.hpp>
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxRoundTransform : public OnnxExtraManager::Transform {
public:
    EXPRP onExecute(EXPRP expr) const override {
        const auto input = expr->inputs()[0];
        const auto floor = _Floor(input);
        const auto zero = _Scalar<float>(0.0f);
        const auto tie = _Equal(_Subtract(input, floor), _Scalar<float>(0.5f));
        const auto even = _Equal(_FloorMod(floor, _Scalar<float>(2.0f)), zero);
        const auto tieValue = _Select(even, floor, _Add(floor, _Scalar<float>(1.0f)));
        // Preserve the sign of zero when a negative tie rounds to zero.
        const auto signedTieValue = _Select(_Equal(tieValue, zero), _Multiply(input, zero), tieValue);
        auto result = _Select(tie, signedTieValue, _Round(input));
        result->setName(expr->name());
        return result->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Round", std::make_shared<OnnxRoundTransform>());
    return true;
}();

} // namespace Express
} // namespace MNN
