#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
namespace MNN {
namespace Express {
static VARP _makeConvForRStep(VARP O, VARP R, int hiddenSize, int direction, VARP Bias) {
    auto rInfo = R->getInfo();
    auto directionStride = rInfo->dim[1] * rInfo->dim[2];
    std::unique_ptr<OpT> matmulOp(new OpT);
    matmulOp->type = OpType_Convolution;
    matmulOp->main.value = new Convolution2DT;
    matmulOp->main.type = OpParameter_Convolution2D;
    matmulOp->main.AsConvolution2D()->common.reset(new Convolution2DCommonT);
    matmulOp->main.AsConvolution2D()->common->outputCount = rInfo->dim[1];
    matmulOp->main.AsConvolution2D()->common->inputCount = rInfo->dim[2];
    matmulOp->main.AsConvolution2D()->weight.resize(rInfo->dim[1] * rInfo->dim[2]);
    ::memcpy(matmulOp->main.AsConvolution2D()->weight.data(), R->readMap<float>() + directionStride * direction, matmulOp->main.AsConvolution2D()->weight.size() * sizeof(float));
    matmulOp->main.AsConvolution2D()->bias.resize(matmulOp->main.AsConvolution2D()->common->outputCount);
    ::memset(matmulOp->main.AsConvolution2D()->bias.data(), 0, matmulOp->main.AsConvolution2D()->bias.size() * sizeof(float));
    if (nullptr != Bias) {
        ::memcpy(matmulOp->main.AsConvolution2D()->bias.data(), Bias->readMap<float>() + direction * matmulOp->main.AsConvolution2D()->bias.size(), matmulOp->main.AsConvolution2D()->bias.size() * sizeof(float));
    }
    auto convX = _Reshape(O, {-1, hiddenSize, 1, 1});
    return Variable::create(Expr::create(matmulOp.get(), {convX}));
}
static VARP _makeConvForW(VARP W, VARP B, VARP X_Input, int inputSize, int direction) {
    auto convX = _Reshape(X_Input, {-1, inputSize, 1, 1});
    auto wInfo = W->getInfo();
    auto directionStride = wInfo->dim[1] * wInfo->dim[2];
    std::unique_ptr<OpT> matmulOp(new OpT);
    matmulOp->type = OpType_Convolution;
    matmulOp->main.value = new Convolution2DT;
    matmulOp->main.type = OpParameter_Convolution2D;
    matmulOp->main.AsConvolution2D()->common.reset(new Convolution2DCommonT);
    matmulOp->main.AsConvolution2D()->common->outputCount = wInfo->dim[1];
    matmulOp->main.AsConvolution2D()->common->inputCount = wInfo->dim[2];
    matmulOp->main.AsConvolution2D()->weight.resize(wInfo->dim[1] * wInfo->dim[2]);
    ::memcpy(matmulOp->main.AsConvolution2D()->weight.data(), W->readMap<float>() + direction * directionStride, matmulOp->main.AsConvolution2D()->weight.size() * sizeof(float));
    matmulOp->main.AsConvolution2D()->bias.resize(matmulOp->main.AsConvolution2D()->common->outputCount);
    if (nullptr != B) {
        ::memcpy(matmulOp->main.AsConvolution2D()->bias.data(), B->readMap<float>() + matmulOp->main.AsConvolution2D()->common->outputCount * direction, matmulOp->main.AsConvolution2D()->bias.size() * sizeof(float));
    } else {
        ::memset(matmulOp->main.AsConvolution2D()->bias.data(), 0, matmulOp->main.AsConvolution2D()->bias.size() * sizeof(float));
    }
    return Variable::create(Expr::create(matmulOp.get(), {convX}));
}
}
};
