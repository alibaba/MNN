#include "../TemplateMerge.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN_generated.h"
#include "config.hpp"
namespace MNN {
namespace Express {
static std::vector <VARP> _UnstackF(VARP value, int axis, int size) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_Unpack;
    MNN_ASSERT(size > 0);
    auto axisParam = new AxisT;
    axisParam->axis = axis;
    op->main.type = OpParameter_Axis;
    op->main.value = axisParam;
    EXPRP expr = Expr::create(std::move(op), {value}, size);
    std::vector<VARP> res;
    for (int i = 0; i < size; ++i) {
        res.emplace_back(Variable::create(expr, i));
    }
    return res;
}
static auto gRegister = []() {
    auto transform = [](EXPRP expr) {
        auto config = Global<modelConfig>::Get();
        if(config->groupConvNative) {
            return false;
        }
        if (expr->get() == nullptr) {
            return false;
        }
        if (expr->get()->type() != OpType_Convolution) {
            return false;
        }
        auto conv2d = expr->get()->main_as_Convolution2D();
        auto common = conv2d->common();
        if (common->group() <= 1) {
            return false;
        }
        if (common->group() == common->inputCount() == common->outputCount()) {
            // Depthwise
            return false;
        }
        if (common->outputCount() / common->group() >= 4 || common->outputCount() / common->group() >= common->group() || common->inputCount() / common->group() >= common->group()) {
            // Large Enough
            return false;
        }
        if (conv2d->weight() == nullptr || conv2d->weight()->data() == nullptr) {
            return false;
        }
        // Currnetly don't support other pad mode
        if (common->padMode() != PadMode_CAFFE) {
            return false;
        }
        // Split As ConvolutionDepthwise
        MNN_ASSERT(conv2d->bias() != nullptr && conv2d->bias()->data() != nullptr);
        auto input = expr->inputs()[0];
        // Input: [b, c, h, w] -> [c/g, b, g, h, w] : [b, c, h, w] -> [b, g, c/g, h, w] -> [c/g, b, g, h, w]
        auto one = _Unsqueeze(_Scalar<int32_t>(1), {0});
        auto negone = _Unsqueeze(_Scalar<int>(-1), {0});
        auto sx = _Shape(input, true);
        auto kernelH = common->kernelY();
        auto kernelW = common->kernelX();
        auto inputChannel = common->inputCount();
        auto outputChannel = common->outputCount();
        auto g = _Unsqueeze(_Scalar<int32_t>(common->group()), {0});
        auto icDivG = common->inputCount() / common->group();
        auto ocDivG = common->outputCount() / common->group();
        auto icdivgv = _Unsqueeze(_Scalar<int32_t>(icDivG), {0});
        auto w = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(3), {0}), one);
        auto h = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(2), {0}), one);
        auto b = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(0), {0}), one);
        
        input = _Reshape(input, _Concat({b, g, icdivgv, h, w}, 0));
        input = _Transpose(input, {2, 0, 1, 3, 4});
        auto inputs = _UnstackF(input, 0, icDivG);
        // Compute Outputs: [c/g, b, g, h, w] -> [oc/g, b, g, oh, ow]
        std::vector<VARP> convMerge(ocDivG);
        for (int y=0; y<ocDivG; ++y) {
            VARP convSummer;
            for (int x=0; x<icDivG; ++x) {
                std::unique_ptr<OpT> op(new OpT);
                op->type = OpType_ConvolutionDepthwise;
                op->main.value = new Convolution2DT;
                op->main.type = OpParameter_Convolution2D;
                op->main.AsConvolution2D()->common.reset(common->UnPack());
                op->main.AsConvolution2D()->common->inputCount = common->group();
                op->main.AsConvolution2D()->common->outputCount = common->group();
                op->main.AsConvolution2D()->common->relu = false;
                op->main.AsConvolution2D()->common->relu6 = false;
                // Copy Bias for the first input
                op->main.AsConvolution2D()->bias = std::vector<float>(common->group(), 0.0f);
                if (x == 0) {
                    for (int j=0; j<common->group(); ++j) {
                        op->main.AsConvolution2D()->bias[j] = conv2d->bias()->data()[ocDivG * j + y];
                    }
                }
                // Copy Weight
                auto kxky = common->kernelX() * common->kernelY();
                op->main.AsConvolution2D()->weight.resize(kxky * common->group());
                for (int j=0; j<common->group(); ++j) {
                    ::memcpy(op->main.AsConvolution2D()->weight.data() + j * kxky, conv2d->weight()->data() + kxky * (j * icDivG * ocDivG + x + y * icDivG), kxky * sizeof(float));
                }
                auto tmp = Variable::create(Expr::create(op.get(), {inputs[x]}));
                if (0 == x) {
                    convSummer = tmp;
                } else {
                    convSummer = convSummer + tmp;
                }
            }
            convMerge[y] = convSummer;
        }
        auto dstFuse = _Stack(convMerge, 0);
        // [oc/g, b, g, oh, ow] -> [b, g, oc/g, oh, ow] -> [b, oc, oh, ow]
        dstFuse = _Transpose(dstFuse, {1, 2, 0, 3, 4});
        auto dx = _Shape(dstFuse, true);
        auto ow = _Slice(dx, _Unsqueeze(_Scalar<int32_t>(4), {0}), one);
        auto oh = _Slice(dx, _Unsqueeze(_Scalar<int32_t>(3), {0}), one);
        dstFuse = _Reshape(dstFuse, _Concat({b, negone, oh, ow}, 0));
        if (common->relu()) {
            dstFuse = _Relu6(dstFuse, 0.0f, 65504.0f);
        } else if (common->relu6()) {
            dstFuse = _Relu6(dstFuse);
        }
        auto groupResult = dstFuse;
        groupResult->setName(expr->outputName(0));
        Expr::replace(expr, groupResult->expr().first);
        return true;
    };
    TemplateMerge::getInstance("Merge").insertTemplateV2("TransformSmallGroupConvolutionToDepthwise", transform);
    return true;
}();

}
} // namespace MNN
